import os
import sys
import threading
import streamlit as st
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import torch
import torch.nn.functional as F
import subprocess
import shutil
import json
import gc
import glob
from transformers import AutoTokenizer, EsmForProteinFolding, AutoModelForMaskedLM
import pyrosetta
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
from Bio.SeqUtils.ProtParam import ProteinAnalysis

_PYROSETTA_INITIALIZED = False

def safe_init_pyrosetta():
    """
    Ensures PyRosetta is initialized exactly once per Streamlit worker thread.
    Required for the cxx11thread build, which uses thread-local singletons.
    """
    # Create a persistent set attached to the sys module to survive Streamlit reloads
    if not hasattr(sys, '_PYROSETTA_INIT_THREADS'):
        sys._PYROSETTA_INIT_THREADS = set()
        
    thread_id = threading.get_ident()
    
    # Check if the current specific thread has initialized PyRosetta
    if thread_id not in sys._PYROSETTA_INIT_THREADS:
        import pyrosetta
        pyrosetta.init("-relax:constrain_relax_to_start_coords -ex1 -ex2 -use_input_sc -mute all")
        sys._PYROSETTA_INIT_THREADS.add(thread_id)

# =====================================================================
# FORMAT-SPECIFIC COMPOSITE SCORE WEIGHTS (Easy to Tune After Benchmarking)
# =====================================================================
FORMAT_WEIGHTS = {
    "scFv (Single-chain Fv)": {
        "camsol_norm": 0.20,
        "janus_norm": 0.20,
        "pyrosetta_norm": 0.20,
        "esm_norm": 0.10,
        "esmc_norm": 0.10,
        "charge_asym_norm": 0.10,
        "pi_norm": 0.10
    },
    "Standard Monoclonal Antibody (IgG / Fab)": {
        "camsol_norm": 0.30,
        "pi_norm": 0.20,
        "gravy_norm": 0.15,
        "janus_norm": 0.15,
        "pyrosetta_norm": 0.10,
        "esm_norm": 0.10
    },
    "scFv-Fc Dimer": {
        "camsol_norm": 0.20,
        "janus_norm": 0.20,
        "pyrosetta_norm": 0.20,
        "charge_asym_norm": 0.15,
        "pi_norm": 0.15,
        "esm_norm": 0.10
    },
    "VHH-Fc / Single-Domain Dimer": {
        "camsol_norm": 0.30,
        "janus_norm": 0.25,
        "pyrosetta_norm": 0.20,
        "esm_norm": 0.15,
        "pi_norm": 0.10
    },
    "Bispecific Fab-scFv": {
        "camsol_norm": 0.35,
        "pi_norm": 0.20,
        "gravy_norm": 0.15,
        "pyrosetta_norm": 0.15,
        "esm_norm": 0.15
    },
    "Bispecific IgG (Asymmetric Heavy/Light Chains)": {
        "camsol_norm": 0.35,
        "pi_norm": 0.20,
        "gravy_norm": 0.15,
        "pyrosetta_norm": 0.15,
        "esm_norm": 0.15
    },
    "Monomeric Enzyme / Non-Antibody": {
        "janus_norm": 0.30,
        "pyrosetta_norm": 0.25,
        "esm_norm": 0.20,
        "esmc_norm": 0.15,
        "camsol_norm": 0.10
    },
    "Custom Multi-Chain Complex": {
        "pyrosetta_norm": 0.30,
        "camsol_norm": 0.25,
        "esm_norm": 0.20,
        "pi_norm": 0.15,
        "gravy_norm": 0.10
    }
}

def parse_custom_fixed_string(custom_str: str, chain_lengths: dict) -> dict:
    """
    Parses user string like 'A: 1-10, 45; B: 15-25' into 1-indexed sets per chain.
    """
    custom_map = {c: set() for c in chain_lengths}
    if not custom_str or not custom_str.strip():
        return custom_map

    for block in custom_str.split(";"):
        block = block.strip()
        if not block or ":" not in block:
            continue
        c_part, ranges_part = block.split(":", 1)
        c_id = c_part.strip().upper()
        if c_id not in custom_map:
            continue
        for token in ranges_part.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                try:
                    s, e = map(int, token.split("-", 1))
                    custom_map[c_id].update(range(max(1, s), min(chain_lengths[c_id], e) + 1))
                except ValueError:
                    pass
            else:
                try:
                    pos = int(token)
                    if 1 <= pos <= chain_lengths[c_id]:
                        custom_map[c_id].add(pos)
                except ValueError:
                    pass
    return custom_map


def build_unified_frozen_positions(
    chains: dict,
    chain_roles: dict,
    frozen_imgt_regions: list,
    custom_fixed_residues: str,
    freeze_native_cys: bool
):
    """
    Returns:
      - fixed_positions_map: dict {chain_id: sorted list of 1-based frozen indices}
      - antifold_mutable_regions_str: space-separated IMGT region string for AntiFold
    """
    chain_lengths = {k: len(v) for k, v in chains.items()}
    custom_map = parse_custom_fixed_string(custom_fixed_residues, chain_lengths)
    fixed_positions_map = {}

    try:
        from abnumber import Chain as AbChain
    except ImportError:
        AbChain = None
        print(" -> WARNING: abnumber not installed. Cannot parse IMGT regions.")

    for c_id, c_seq in chains.items():
        role = chain_roles.get(c_id, "")
        frozen_set = set(custom_map.get(c_id, set()))

        # 1. Freeze Native Cysteines if enabled
        if freeze_native_cys:
            for idx_0, aa in enumerate(c_seq):
                if aa == "C":
                    frozen_set.add(idx_0 + 1)

        # 2. Freeze user-selected IMGT regions on antibody chains
        if role in ["VH", "VL", "VHH"] and AbChain is not None and frozen_imgt_regions:
            try:
                ab = AbChain(c_seq, scheme='imgt')
                region_seq_map = {
                    "FR1": ab.fr1_seq,
                    "CDR1": ab.cdr1_seq,
                    "FR2": ab.fr2_seq,
                    "CDR2": ab.cdr2_seq,
                    "FR3": ab.fr3_seq,
                    "CDR3": ab.cdr3_seq,
                    "FR4": ab.fr4_seq
                }
                # Track cumulative offset across contiguous IMGT regions for exact indexing
                search_start = 0
                for reg_name in ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]:
                    r_seq = region_seq_map.get(reg_name, "")
                    if r_seq:
                        pos = c_seq.find(r_seq, search_start)
                        if pos != -1:
                            if reg_name in frozen_imgt_regions:
                                frozen_set.update(range(pos + 1, pos + 1 + len(r_seq)))
                            search_start = pos + len(r_seq)
            except Exception as e:
                print(f" -> WARNING: abnumber parsing failed for Chain {c_id}: {e}")

        fixed_positions_map[c_id] = sorted(list(frozen_set))

    # Build matching AntiFold --regions string (AntiFold expects MUTABLE regions)
    all_imgt = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]
    mutable_imgt = [r for r in all_imgt if r not in (frozen_imgt_regions or [])]
    antifold_tokens = []
    for r in mutable_imgt:
        if r.startswith("FR"):
            idx_num = r[-1]
            antifold_tokens.extend([f"FWH{idx_num}", f"FWL{idx_num}"])
        elif r.startswith("CDR"):
            idx_num = r[-1]
            antifold_tokens.extend([f"CDRH{idx_num}", f"CDRL{idx_num}"])
    antifold_mutable_regions_str = " ".join(antifold_tokens) if antifold_tokens else "FWL FWH"

    return fixed_positions_map, antifold_mutable_regions_str


def enforce_mutation_cap_and_filters(candidate_seq: str, wt_seq: str, global_frozen_1idx: set, omit_aas: list, max_muts_per_seq: int, rng: np.random.Generator) -> str:
    """
    1. Reverts any mutation at a frozen position or into a banned amino acid (omit_aas).
    2. If the remaining mutation count exceeds max_muts_per_seq (e.g., from AntiFold/MPNN),
       subsamples to a target count distributed uniformly between 1 and max_muts_per_seq.
    """
    if len(candidate_seq) != len(wt_seq):
        return wt_seq

    cand_list = list(candidate_seq)
    omit_set = set(omit_aas or [])
    valid_diff_indices = []

    for i, (wt_aa, mut_aa) in enumerate(zip(wt_seq, cand_list)):
        pos_1idx = i + 1
        if wt_aa != mut_aa:
            if pos_1idx in global_frozen_1idx or mut_aa in omit_set:
                cand_list[i] = wt_aa
            else:
                valid_diff_indices.append(i)

    # Spread multi-mutant sequences evenly across [1 .. max_muts_per_seq]
    if len(valid_diff_indices) > max_muts_per_seq:
        target_k = int(rng.integers(1, max_muts_per_seq + 1))
        keep_indices = set(rng.choice(valid_diff_indices, size=target_k, replace=False))
        for i in valid_diff_indices:
            if i not in keep_indices:
                cand_list[i] = wt_seq[i]

    return "".join(cand_list)

def build_single_and_combo_records(sorted_single_muts: list, clean_wt_full: str, model_prefix: str, num_seqs: int, max_muts_per_seq: int, rng: np.random.Generator) -> list:
    """
    Used by ThermoMPNN, ESM-C, and AbLang2:
      - First half of num_seqs: top single-point mutations (1 mutation each)
      - Second half of num_seqs: combinations of 2..max_muts_per_seq non-overlapping top mutations
    """
    records = []
    if not sorted_single_muts:
        return records

    n_singles = num_seqs if max_muts_per_seq <= 1 else max(1, num_seqs // 2)
    n_combos = max(0, num_seqs - n_singles)

    # 1. Top Single Mutants (1 mutation)
    for i, m in enumerate(sorted_single_muts[:n_singles]):
        pos = m.get('global_pos', m.get('pos'))
        mut_seq = list(clean_wt_full)
        if pos is not None and 0 <= pos < len(mut_seq):
            mut_seq[pos] = m['mut_aa']
            mut_tag = f"{m['wt_aa']}{pos+1}{m['mut_aa']}"
            records.append(SeqRecord(
                Seq("".join(mut_seq)),
                id=f"{model_prefix}_{mut_tag}_rank{i+1}",
                description=""
            ))

    # 2. Multi-Mutant Combinations (2 to max_muts_per_seq mutations from top pool)
    best_per_pos = []
    seen_pos = set()
    for m in sorted_single_muts:
        pos = m.get('global_pos', m.get('pos'))
        if pos is not None and pos not in seen_pos:
            seen_pos.add(pos)
            best_per_pos.append(m)

    pool = best_per_pos[:min(len(best_per_pos), 30)]  # Combine from top 30 mutable positions
    if n_combos > 0 and len(pool) >= 2:
        max_k = min(max_muts_per_seq, len(pool))
        for c_idx in range(n_combos):
            k = int(rng.integers(2, max_k + 1))
            chosen_indices = rng.choice(len(pool), size=k, replace=False)
            chosen = sorted([pool[idx] for idx in chosen_indices], key=lambda x: x.get('global_pos', x.get('pos')))
            
            mut_seq = list(clean_wt_full)
            tags = []
            for m in chosen:
                pos = m.get('global_pos', m.get('pos'))
                mut_seq[pos] = m['mut_aa']
                tags.append(f"{m['wt_aa']}{pos+1}{m['mut_aa']}")
                
            records.append(SeqRecord(
                Seq("".join(mut_seq)),
                id=f"{model_prefix}_combo{k}mut_{'-'.join(tags)}_c{c_idx+1}",
                description=""
            ))

    return records

def run_universal_developability_pipeline(
    chains: dict, 
    chain_roles: dict, 
    base_output_dir: str, 
    run_name: str, 
    protein_format: str,
    gen_models: list = None, 
    num_seqs: int = 100, 
    temp: float = 0.2,
    frozen_imgt_regions: list = None,
    custom_fixed_residues: str = "",
    freeze_native_cys: bool = True,
    omit_aas: list = None,
    max_muts_per_seq: int = 5,
    mpnn_backbone_noise: float = 0.02,
    thermo_ddg_cutoff: float = -0.3
):
    """
    Universal Phase 1 pipeline supporting N-chain complexes, antibodies, and enzymes.
    - chains: dict of sequence strings {'A': '...', 'B': '...'}
    - chain_roles: dict mapping chain keys to roles {'A': 'VH', 'B': 'VL'} or {'A': 'Chain_A'}
    """
    if gen_models is None:
        gen_models = []
    if frozen_imgt_regions is None:
        frozen_imgt_regions = ["CDR1", "CDR2", "CDR3"]
    if omit_aas is None:
        omit_aas = ["C"]

    run_dir = os.path.join(base_output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    unrelaxed_pdb_path = os.path.join(run_dir, f"{run_name}_unrelaxed.pdb")
    relaxed_pdb_path = os.path.join(run_dir, f"{run_name}_relaxed.pdb")
    metadata_path = os.path.join(run_dir, f"{run_name}_metadata.json")

    # Compute unified frozen positions across all chains
    fixed_positions_map, antifold_regions_str = build_unified_frozen_positions(
        chains=chains,
        chain_roles=chain_roles,
        frozen_imgt_regions=frozen_imgt_regions,
        custom_fixed_residues=custom_fixed_residues,
        freeze_native_cys=freeze_native_cys
    )

    # Convert per-chain 1-indexed frozen positions into global concatenated 1-indexed set
    global_frozen_1idx = set()
    offset = 0
    for c_id, c_seq in chains.items():
        for local_pos in fixed_positions_map.get(c_id, []):
            global_frozen_1idx.add(offset + local_pos)
        offset += len(c_seq)

    # ---------------------------------------------------------
    # STEP 1: Save Run Manifest & Metadata
    # ---------------------------------------------------------
    chain_lengths = {k: len(v) for k, v in chains.items()}
    run_metadata = {
        "protein_format": protein_format,
        "chain_lengths": chain_lengths,
        "chain_roles": chain_roles,
        "frozen_imgt_regions": frozen_imgt_regions,
        "custom_fixed_residues": custom_fixed_residues,
        "freeze_native_cys": freeze_native_cys,
        "omit_aas": omit_aas,
        "max_muts_per_seq": max_muts_per_seq,
        "mpnn_backbone_noise": mpnn_backbone_noise,
        "thermo_ddg_cutoff": thermo_ddg_cutoff,
        "fixed_positions_map": fixed_positions_map
    }
    with open(metadata_path, "w") as f:
        json.dump(run_metadata, f, indent=4)
    print(f" -> Saved run metadata. Frozen residue counts per chain: "
          f"{ {k: len(v) for k, v in fixed_positions_map.items()} }")

    # ---------------------------------------------------------
    # STEP 2: N-Chain Structure Prediction (ESMFold Multimer Fix)
    # ---------------------------------------------------------
    if os.path.exists(unrelaxed_pdb_path):
        print(f"Step 2: Found existing {unrelaxed_pdb_path}. Skipping ESMFold prediction...")
    else:
        print("Step 2: Running Universal Multi-Chain Structure Prediction...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.trunk.set_chunk_size(64)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Concatenate all chains into a single continuous sequence for ESMFold
        combined_seq = "".join(chains.values())
        
        tokenized_input = tokenizer(
            combined_seq, 
            return_tensors="pt", 
            add_special_tokens=False, 
            return_token_type_ids=False
        ).to(device)
        
        # Apply cumulative +512 position ID offset at each chain boundary
        seq_length = len(combined_seq)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        
        current_idx = 0
        for i, (chain_id, seq) in enumerate(chains.items()):
            if i > 0:
                position_ids[current_idx:] += 512
            current_idx += len(seq)
            
        tokenized_input["position_ids"] = position_ids.unsqueeze(0)
        
        with torch.no_grad():
            output = model(**tokenized_input)
            float32_output = {
                k: v.to(torch.float32) if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16 else v 
                for k, v in output.items()
            }
            pdb_string = model.output_to_pdb(float32_output)[0]
            
        # Reassign PDB chain IDs and local residue numbering (1 to L per chain)
        pdb_lines = pdb_string.split("\n")
        fixed_pdb = []
        
        # Precompute the exact ESMFold index bounds (including the +512 gap)
        chain_bounds = []
        current_offset = 0
        for c_id, c_seq in chains.items():
            start_idx_in_pdb = current_offset + 1
            end_idx_in_pdb = start_idx_in_pdb + len(c_seq) - 1
            chain_bounds.append((c_id, start_idx_in_pdb, end_idx_in_pdb))
            current_offset += len(c_seq) + 512  # The +512 gap MUST be tracked here
            
        previous_chain = chain_bounds[0][0]
        
        for line in pdb_lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                res_idx_global = int(line[22:26].strip())
                
                target_chain = chain_bounds[0][0]
                local_res = 1
                
                # Map the raw ESMFold index to the correct chain and local 1-index
                for c_id, start_pdb, end_pdb in chain_bounds:
                    if start_pdb <= res_idx_global <= end_pdb:
                        target_chain = c_id
                        local_res = res_idx_global - start_pdb + 1
                        break
                
                # Format line with correct chain letter and 1-indexed local residue number
                line = line[:21] + target_chain + line[22:]
                line = line[:22] + f"{local_res:4d}" + line[26:]
                
                if target_chain != previous_chain:
                    fixed_pdb.append("TER")
                    previous_chain = target_chain
                    
            fixed_pdb.append(line)
            
        with open(unrelaxed_pdb_path, "w") as f:
            f.write("\n".join(fixed_pdb))
            
        del model
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # STEP 3: Constrained Relaxation (PyRosetta)
    # ---------------------------------------------------------
    if os.path.exists(relaxed_pdb_path):
        print(f"Step 3: Found existing {relaxed_pdb_path}. Skipping PyRosetta relaxation...")
    else:
        print("Step 3: Running Constrained PyRosetta FastRelax...")
        safe_init_pyrosetta()
        import pyrosetta
        from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
        from pyrosetta.rosetta.protocols.relax import FastRelax
        
        pose = pyrosetta.pose_from_file(unrelaxed_pdb_path)
        scorefxn = ScoreFunctionFactory.create_score_function("ref2015")
        
        relax = FastRelax()
        relax.set_scorefxn(scorefxn)
        relax.constrain_relax_to_start_coords(True)
        relax.apply(pose)
        
        pose.dump_pdb(relaxed_pdb_path)
        
        # Free C++ memory
        pose.clear()
        del pose


    # ---------------------------------------------------------
    # STEP 4: Generative Ensemble (Unified Masking & Settings)
    # ---------------------------------------------------------
    print(f"Step 4: Running Generative Ensemble: {gen_models}")
    all_raw_records = []
    rng = np.random.default_rng(seed=42)
    
    clean_wt_full = "".join(chains.values())

    # Dynamic chain strings for N-chain compatibility (1-chain VHH/Enzyme, 2-chain Fv, 4-chain Bispecific)
    mpnn_chains_str = " ".join(chains.keys())    # e.g., "A", "A B", or "A B C D"
    thermo_chains_str = "".join(chains.keys())   # e.g., "A", "AB", or "ABCD"
    combined_frozen_0idx = {p - 1 for p in global_frozen_1idx}

    # Build a list of antibody design units:
    #   - Paired Fv arm: (vh_chain_id, vl_chain_id)
    #   - Single-domain VHH/unpaired VH arm: (vhh_chain_id, None)
    vh_chains = [k for k, v in chain_roles.items() if v == 'VH']
    vl_chains = [k for k, v in chain_roles.items() if v == 'VL']
    vhh_chains = [k for k, v in chain_roles.items() if v == 'VHH']

    antibody_units = []
    for idx_pair in range(min(len(vh_chains), len(vl_chains))):
        antibody_units.append((vh_chains[idx_pair], vl_chains[idx_pair]))
    for extra_vh in vh_chains[len(vl_chains):]:
        antibody_units.append((extra_vh, None))
    for vhh_c in vhh_chains:
        antibody_units.append((vhh_c, None))

    # Precompute 0-based global chain offsets for multi-chain splicing
    chain_offsets, curr_off = {}, 0
    for c_k, c_s in chains.items():
        chain_offsets[c_k] = curr_off
        curr_off += len(c_s)

    def splice_chains_into_full_complex(chain_updates: dict) -> str:
        return "".join([chain_updates.get(c_key, wt_c_seq) for c_key, wt_c_seq in chains.items()])

    # --- MODEL A: AntiFold (Supports Paired VH/VL and Single-Domain VHH) ---
    if "AntiFold (Antibody-specific)" in gen_models:
        if not antibody_units:
            print(" -> WARNING: AntiFold requires at least one VH/VL pair or VHH chain. Skipping AntiFold.")
        else:
            af_records = []
            sample_idx = 1
            for ch_heavy, ch_light in antibody_units:
                len_h = len(chains[ch_heavy])
                
                if ch_light is not None:
                    print(f" -> Running AntiFold (Paired Mode: {ch_heavy}=VH, {ch_light}=VL), regions: {antifold_regions_str}")
                    antifold_cmd = [
                        "python", "-m", "antifold.main",
                        "--pdb_file", relaxed_pdb_path,
                        "--heavy_chain", ch_heavy, "--light_chain", ch_light,
                        "--regions", antifold_regions_str,
                        "--num_seq_per_target", str(num_seqs),
                        "--sampling_temp", str(temp),
                        "--out_dir", run_dir
                    ]
                    expected_fastas = [
                        os.path.join(run_dir, f"{run_name}_relaxed_{ch_heavy}{ch_light}.fasta"),
                        os.path.join(run_dir, f"{run_name}_relaxed_AB.fasta")
                    ]
                else:
                    # Single-Domain VHH / Nanobody Mode (--nanobody_chain)
                    # Filter regions string to only heavy-chain IMGT tokens (FWH*, CDRH*)
                    vhh_regions = " ".join([t for t in antifold_regions_str.split() if "H" in t]) or "FWH1 FWH2 FWH3 FWH4"
                    print(f" -> Running AntiFold (Nanobody/VHH Mode: Chain {ch_heavy}), regions: {vhh_regions}")
                    antifold_cmd = [
                        "python", "-m", "antifold.main",
                        "--pdb_file", relaxed_pdb_path,
                        "--nanobody_chain", ch_heavy,
                        "--regions", vhh_regions,
                        "--num_seq_per_target", str(num_seqs),
                        "--sampling_temp", str(temp),
                        "--out_dir", run_dir
                    ]
                    expected_fastas = [
                        os.path.join(run_dir, f"{run_name}_relaxed_{ch_heavy}.fasta"),
                        os.path.join(run_dir, f"{run_name}_relaxed_A.fasta")
                    ]

                try:
                    subprocess.run(antifold_cmd, check=True, capture_output=True, text=True)
                    af_fasta = next((p for p in expected_fastas if os.path.exists(p)), None)
                    if af_fasta is None:
                        # Fallback glob for any fasta just written by AntiFold
                        af_candidates = glob.glob(os.path.join(run_dir, f"{run_name}_relaxed_*.fasta"))
                        af_fasta = af_candidates[0] if af_candidates else None

                    if af_fasta and os.path.exists(af_fasta):
                        for rec in SeqIO.parse(af_fasta, "fasta"):
                            s_clean = str(rec.seq).replace(":", "").replace("/", "")
                            if ch_light is not None and len(s_clean) == (len_h + len(chains[ch_light])):
                                s_full = splice_chains_into_full_complex({
                                    ch_heavy: s_clean[:len_h],
                                    ch_light: s_clean[len_h:]
                                })
                            elif ch_light is None and len(s_clean) == len_h:
                                s_full = splice_chains_into_full_complex({ch_heavy: s_clean})
                            else:
                                s_full = s_clean

                            if s_full == clean_wt_full:
                                continue
                            af_records.append(SeqRecord(Seq(s_full), id=f"AntiFold_sample{sample_idx}", description=""))
                            sample_idx += 1
                        os.remove(af_fasta)
                except subprocess.CalledProcessError as e:
                    print(f" -> WARNING: AntiFold failed on chain {ch_heavy}: {e.stderr}")

            if af_records:
                SeqIO.write(af_records, os.path.join(run_dir, f"{run_name}_antifold.fasta"), "fasta")
                all_raw_records.extend(af_records)

    # --- EXPORT UNIFIED MASKING JSONL FOR PROTEINMPNN / THERMOMPNN ---
    fixed_json_path = os.path.join(run_dir, "fixed_positions.jsonl")
    has_mpnn = any("ProteinMPNN" in m or "ThermoMPNN" in m for m in gen_models)
    
    if has_mpnn:
        pdb_basename = os.path.splitext(os.path.basename(relaxed_pdb_path))[0]
        fixed_dict = {pdb_basename: fixed_positions_map}
        with open(fixed_json_path, 'w') as f:
            f.write(json.dumps(fixed_dict) + "\n")
        print(f" -> Generated unified MPNN masking file at {fixed_json_path}")

    # --- MODEL B: ProteinMPNN (Diverse) ---
    if "ProteinMPNN (Diverse)" in gen_models:
        print(f" -> Running ProteinMPNN at Temperature {temp} on chains ({mpnn_chains_str})...")
        mpnn_out_dir = os.path.join(run_dir, "mpnn_out")
        PROTEIN_MPNN_SCRIPT = "/home/esigalas/workspace/ProteinMPNN/protein_mpnn_run.py"
        
        if not os.path.exists(PROTEIN_MPNN_SCRIPT):
            st.error(f"Could not find ProteinMPNN at {PROTEIN_MPNN_SCRIPT}.")
            raise FileNotFoundError("Missing ProteinMPNN script.")
            
        mpnn_cmd = [
            "python", PROTEIN_MPNN_SCRIPT,
            "--pdb_path", relaxed_pdb_path,
            "--pdb_path_chains", mpnn_chains_str,
            "--fixed_positions_jsonl", fixed_json_path,
            "--out_folder", mpnn_out_dir,
            "--num_seq_per_target", str(num_seqs),
            "--sampling_temp", str(temp),
            "--batch_size", "1",
            "--backbone_noise", str(mpnn_backbone_noise)
        ]
        if omit_aas:
            mpnn_cmd.extend(["--omit_AAs", "".join(omit_aas)])

        try:
            subprocess.run(mpnn_cmd, check=True, capture_output=True, text=True)
            mpnn_fastas = glob.glob(os.path.join(mpnn_out_dir, "seqs", f"{pdb_basename}.fa"))
            mpnn_records = []
            sample_idx = 1
            for fa in mpnn_fastas:
                for rec in SeqIO.parse(fa, "fasta"):
                    s_clean = str(rec.seq).replace(":", "").replace("/", "")
                    # Skip the 0-th record in ProteinMPNN .fa files (which is the native WT sequence)
                    if s_clean == clean_wt_full or rec.id.startswith(run_name):
                        continue
                    rec.id = f"ProteinMPNN_sample{sample_idx}"
                    rec.description = ""
                    mpnn_records.append(rec)
                    sample_idx += 1
                    
            if mpnn_records:
                SeqIO.write(mpnn_records, os.path.join(run_dir, f"{run_name}_proteinmpnn.fasta"), "fasta")
                all_raw_records.extend(mpnn_records)
        except subprocess.CalledProcessError as e:
            st.error(f"**ProteinMPNN Crashed!**\n\nError Log:\n```\n{e.stderr}\n```")
            raise RuntimeError("Phase 1 halted due to ProteinMPNN failure.")

    # --- MODEL C: ThermoMPNN (ddG-Driven Single + Multi-Mutant Combos) ---
    if "ThermoMPNN (ddG-Driven)" in gen_models:
        print(f" -> Running ThermoMPNN ddG Generator on chains ({thermo_chains_str})...")
        THERMO_MPNN_SCRIPT = "/home/esigalas/workspace/ThermoMPNN/analysis/custom_inference.py"
        
        if not os.path.exists(THERMO_MPNN_SCRIPT):
            st.error(f"Could not find ThermoMPNN at {THERMO_MPNN_SCRIPT}.")
            st.stop()
            
        pdb_basename = os.path.splitext(os.path.basename(relaxed_pdb_path))[0]
        csv_path = os.path.join(run_dir, f"ThermoMPNN_inference_{pdb_basename}.csv")
        
        all_thermo_muts = []
        omit_set = set(omit_aas or [])
        
        thermo_cmd = [
            "python", THERMO_MPNN_SCRIPT,
            "--pdb", relaxed_pdb_path,
            "--chain", thermo_chains_str, 
            "--out_dir", run_dir
        ]
        try:
            subprocess.run(thermo_cmd, check=True, capture_output=True, text=True)
            if not os.path.exists(csv_path):
                st.error("ThermoMPNN failed to create output CSV.")
                st.stop()
            df_thermo = pd.read_csv(csv_path)
            for _, row in df_thermo.iterrows():
                pos = int(row['position']) 
                mut_aa = str(row['mutation']).strip()
                ddg = float(row['ddG_pred'])
                
                # Safeguard if ThermoMPNN CSV uses local per-chain position instead of global 0-index
                if 'chain' in df_thermo.columns and str(row['chain']).strip() in chain_offsets:
                    c_key = str(row['chain']).strip()
                    if (0 <= pos < len(clean_wt_full)) and clean_wt_full[pos] != wt_aa:
                        cand_global = chain_offsets[c_key] + pos
                        if 0 <= cand_global < len(clean_wt_full) and clean_wt_full[cand_global] == wt_aa:
                            pos = cand_global

                if pos in combined_frozen_0idx or mut_aa in omit_set:
                    continue 
                if ddg < thermo_ddg_cutoff:
                    all_thermo_muts.append({
                        'pos': pos, 'wt_aa': row['wildtype'], 'mut_aa': mut_aa, 'ddg': ddg
                    })
            os.remove(csv_path)
        except subprocess.CalledProcessError as e:
            st.error(f"**ThermoMPNN Crashed!**\n\nError Log:\n```\n{e.stderr}\n```")
            st.stop()
            
        # Sort by most stabilizing (most negative ddG)
        all_thermo_muts = sorted(all_thermo_muts, key=lambda x: x['ddg'])
        thermo_records = build_single_and_combo_records(
            sorted_single_muts=all_thermo_muts,
            clean_wt_full=clean_wt_full,
            model_prefix="ThermoMPNN",
            num_seqs=int(num_seqs),
            max_muts_per_seq=int(max_muts_per_seq),
            rng=rng
        )
        if thermo_records:
            SeqIO.write(thermo_records, os.path.join(run_dir, f"{run_name}_thermompnn.fasta"), "fasta")
            all_raw_records.extend(thermo_records)

    # --- MODEL D: ProteinMPNN (Soluble) ---
    if "ProteinMPNN (Soluble)" in gen_models:
        print(f" -> Running ProteinMPNN (Soluble Model) at Temperature {temp} on chains ({mpnn_chains_str})...")
        soluble_out_dir = os.path.join(run_dir, "soluble_mpnn_out")
        PROTEIN_MPNN_SCRIPT = "/home/esigalas/workspace/ProteinMPNN/protein_mpnn_run.py"
        
        soluble_cmd = [
            "python", PROTEIN_MPNN_SCRIPT,
            "--pdb_path", relaxed_pdb_path,
            "--pdb_path_chains", mpnn_chains_str,
            "--fixed_positions_jsonl", fixed_json_path,
            "--out_folder", soluble_out_dir,
            "--num_seq_per_target", str(num_seqs),
            "--sampling_temp", str(temp),
            "--batch_size", "1",
            "--use_soluble_model",
            "--backbone_noise", str(mpnn_backbone_noise)
        ]
        if omit_aas:
            soluble_cmd.extend(["--omit_AAs", "".join(omit_aas)])

        try:
            subprocess.run(soluble_cmd, check=True, capture_output=True, text=True)
            soluble_fastas = glob.glob(os.path.join(soluble_out_dir, "seqs", "*.fa"))
            soluble_records = []
            sample_idx = 1
            for fa in soluble_fastas:
                for rec in SeqIO.parse(fa, "fasta"):
                    s_clean = str(rec.seq).replace(":", "").replace("/", "")
                    # Skip the 0-th native WT record in ProteinMPNN output
                    if s_clean == clean_wt_full or rec.id.startswith(run_name):
                        continue
                    rec.id = f"SolubleMPNN_sample{sample_idx}"
                    rec.description = ""
                    soluble_records.append(rec)
                    sample_idx += 1
                    
            if soluble_records:
                SeqIO.write(soluble_records, os.path.join(run_dir, f"{run_name}_solublempnn.fasta"), "fasta")
                all_raw_records.extend(soluble_records)
        except subprocess.CalledProcessError as e:
            st.error(f"**ProteinMPNN (Soluble) Crashed!**\n\nError Log:\n```\n{e.stderr}\n```")
            st.stop()

    # --- MODEL E: ESM-C (Single + Multi-Mutant Combos) ---
    if "ESM-C (Evolutionary LM)" in gen_models:
        print(" -> Running ESM-C Masked-Marginal Probability Generator (Isolated Env)...")
        esmc_fasta = os.path.join(run_dir, f"{run_name}_esmc.fasta")
        cdr_str = ",".join(map(str, sorted(list(combined_frozen_0idx))))
        omit_set = set(omit_aas or [])
        
        esmc_cmd = [
            "conda", "run", "-n", "esmc_env", 
            "python", "run_esmc_dms.py",
            "--sequence", clean_wt_full,
            "--cdr_indices", cdr_str,
            "--num_seqs", str(num_seqs),
            "--out_fasta", esmc_fasta
        ]
        try:
            subprocess.run(esmc_cmd, check=True, capture_output=True, text=True)
            if os.path.exists(esmc_fasta):
                raw_esmc = list(SeqIO.parse(esmc_fasta, "fasta"))
                
                # Extract ranked single point mutations from the ESM-C DMS output
                esmc_single_muts = []
                for rec in raw_esmc:
                    s_clean = str(rec.seq).strip().replace(":", "").replace("/", "")
                    if len(s_clean) != len(clean_wt_full):
                        continue
                    diffs = [
                        (idx, clean_wt_full[idx], s_clean[idx])
                        for idx in range(len(clean_wt_full))
                        if s_clean[idx] != clean_wt_full[idx]
                    ]
                    # Keep valid single-point mutations that respect frozen indices and banned AAs
                    if len(diffs) == 1:
                        pos, wt_aa, mut_aa = diffs[0]
                        if pos not in combined_frozen_0idx and mut_aa not in omit_set:
                            esmc_single_muts.append({
                                'pos': pos, 'wt_aa': wt_aa, 'mut_aa': mut_aa
                            })

                # Build 50% single mutants and 50% multi-mutant combinations (2..max_muts_per_seq)
                esmc_records = build_single_and_combo_records(
                    sorted_single_muts=esmc_single_muts,
                    clean_wt_full=clean_wt_full,
                    model_prefix="ESMC",
                    num_seqs=int(num_seqs),
                    max_muts_per_seq=int(max_muts_per_seq),
                    rng=rng
                )
                if esmc_records:
                    SeqIO.write(esmc_records, esmc_fasta, "fasta")
                    all_raw_records.extend(esmc_records)
        except subprocess.CalledProcessError as e:
            st.error(f"**ESM-C Subprocess Crashed!**\n\nError Log:\n```\n{e.stderr}\n```")
            st.stop()

    # --- MODEL F: AbLang2 (Supports Paired VH/VL & Single-Domain VHH, Single + Combos) ---
    if "AbLang2 (Antibody LM)" in gen_models:
        if not antibody_units:
            print(" -> WARNING: AbLang2 requires at least one VH, VL, or VHH chain. Skipping AbLang2.")
        else:
            print(f" -> Running AbLang2 Masked-Marginal Generator across {len(antibody_units)} antibody unit(s)...")
            import ablang2
            try:
                ablang = ablang2.pretrained(model_to_use="ablang2-paired", random_init=False)
                alphabet = list("ACDEFGHIKLMNPQRSTVWY")
                omit_set = set(omit_aas or [])
                ablang_muts = []

                for ch_heavy, ch_light in antibody_units:
                    h_seq = chains[ch_heavy]
                    l_seq = chains[ch_light] if ch_light is not None else ""
                    frozen_h_0idx = {p - 1 for p in fixed_positions_map.get(ch_heavy, [])}
                    h_len = len(h_seq)

                    # 1. Scan Heavy / VHH Chain
                    for i in range(h_len):
                        if i in frozen_h_0idx:
                            continue
                        wt_aa = h_seq[i]
                        masked_vh = h_seq[:i] + "*" + h_seq[i+1:]
                        res = ablang([[masked_vh, l_seq]], mode='probability')
                        probs = res[0][i]
                        wt_idx = alphabet.index(wt_aa) if wt_aa in alphabet else None
                        if wt_idx is None:
                            continue
                        wt_prob = probs[wt_idx]
                        for mut_idx, mut_aa in enumerate(alphabet):
                            if mut_aa == wt_aa or mut_aa in omit_set:
                                continue
                            delta_p = probs[mut_idx] - wt_prob
                            if delta_p > 0:
                                ablang_muts.append({
                                    'global_pos': chain_offsets[ch_heavy] + i,
                                    'wt_aa': wt_aa, 'mut_aa': mut_aa, 'delta': delta_p
                                })

                    # 2. Scan Light Chain (if paired VL exists)
                    if ch_light is not None:
                        frozen_l_0idx = {p - 1 for p in fixed_positions_map.get(ch_light, [])}
                        for i in range(len(l_seq)):
                            if i in frozen_l_0idx:
                                continue
                            wt_aa = l_seq[i]
                            masked_vl = l_seq[:i] + "*" + l_seq[i+1:]
                            res = ablang([[h_seq, masked_vl]], mode='probability')
                            probs = res[0][h_len + i]
                            wt_idx = alphabet.index(wt_aa) if wt_aa in alphabet else None
                            if wt_idx is None:
                                continue
                            wt_prob = probs[wt_idx]
                            for mut_idx, mut_aa in enumerate(alphabet):
                                if mut_aa == wt_aa or mut_aa in omit_set:
                                    continue
                                delta_p = probs[mut_idx] - wt_prob
                                if delta_p > 0:
                                    ablang_muts.append({
                                        'global_pos': chain_offsets[ch_light] + i,
                                        'wt_aa': wt_aa, 'mut_aa': mut_aa, 'delta': delta_p
                                    })

                ablang_muts = sorted(ablang_muts, key=lambda x: x['delta'], reverse=True)
                ablang_records = build_single_and_combo_records(
                    sorted_single_muts=ablang_muts,
                    clean_wt_full=clean_wt_full,
                    model_prefix="AbLang2Paired",
                    num_seqs=int(num_seqs),
                    max_muts_per_seq=int(max_muts_per_seq),
                    rng=rng
                )
                if ablang_records:
                    SeqIO.write(ablang_records, os.path.join(run_dir, f"{run_name}_ablang2.fasta"), "fasta")
                    all_raw_records.extend(ablang_records)
            except Exception as e:
                st.error(f"**AbLang2 Crashed!**\n\nError Log:\n```\n{str(e)}\n```")
                st.stop()

    # ---------------------------------------------------------
    # STEP 5: Post-Filtering (Mask Enforcement, Mutation Cap) & Deduplication
    # ---------------------------------------------------------
    camsol_fasta_path = os.path.join(run_dir, f"{run_name}_camsol_upload.fasta")
    print(f"Step 5: Enforcing frozen regions, max_muts={max_muts_per_seq}, and pooling -> {camsol_fasta_path}")
    
    wt_record = SeqRecord(Seq(clean_wt_full), id=f"{run_name}_WT", description="")
    cleaned_records = [wt_record]
    
    # Map clean_seq -> SeqRecord so we can track or merge duplicate hits across models
    seq_to_record = {clean_wt_full: wt_record}
    duplicate_count = 0
    
    for record in all_raw_records:
        raw_seq = str(record.seq).replace(":", "").replace("/", "")
        # Enforce exact user-defined frozen positions, banned AAs, and max mutation count
        filtered_seq = enforce_mutation_cap_and_filters(
            candidate_seq=raw_seq,
            wt_seq=clean_wt_full,
            global_frozen_1idx=global_frozen_1idx,
            omit_aas=omit_aas,
            max_muts_per_seq=max_muts_per_seq,
            rng=rng
        )
        
        if filtered_seq not in seq_to_record:
            new_rec = SeqRecord(Seq(filtered_seq), id=record.id, description="")
            seq_to_record[filtered_seq] = new_rec
            cleaned_records.append(new_rec)
        else:
            existing_rec = seq_to_record[filtered_seq]
            if existing_rec.id != f"{run_name}_WT":
                duplicate_count += 1
                # Extract model prefix (e.g., 'AbLang2Paired' from 'AbLang2Paired_A45G_rank1')
                new_model_tag = record.id.split("_")[0]
                if new_model_tag not in existing_rec.id:
                    existing_rec.id = f"{existing_rec.id}+{new_model_tag}"

    SeqIO.write(cleaned_records, camsol_fasta_path, "fasta")
    print(f"Step 5 Complete: Pooled {len(cleaned_records)} unique variants "
          f"({duplicate_count} duplicates collapsed; cross-model hits tagged with '+').")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def score_universal_variants(run_dir: str, run_name: str, wt_sequence: str):
    """
    Universal Phase 2 scoring engine handling N-chain biophysics, PyRosetta,
    per-domain PLMs, and the Agnostic Gated + Invariant developability score.
    """
    print(f"--- Phase 2: Scoring Universal Variants for {run_name} ---")
    
    final_csv_path = os.path.join(run_dir, f"{run_name}_developability_metrics.csv")
    metadata_path = os.path.join(run_dir, f"{run_name}_metadata.json")
    raw_fasta_path = os.path.join(run_dir, f"{run_name}_camsol_upload.fasta")
    camsol_txt_path = os.path.join(run_dir, f"{run_name}_camsol.txt")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata missing for {run_name}. Re-run Phase 1.")
        
    with open(metadata_path, "r") as f:
        meta = json.load(f)
        chain_lengths = meta["chain_lengths"]
        chain_roles = meta["chain_roles"]
        protein_format = meta.get("protein_format", "scFv (Single-chain Fv)")

    records = list(SeqIO.parse(raw_fasta_path, "fasta"))
    clean_wt = wt_sequence.replace(":", "").replace("/", "")
    wt_list = list(clean_wt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Precompute 0-based global chain bounds and antibody design units (supports 2-chain Fv, VHH, 4-chain Bispecifics)
    chain_bounds = {}  # {c_id: (start_0idx, end_0idx, role)}
    curr_idx = 0
    for c_id, c_len in chain_lengths.items():
        chain_bounds[c_id] = (curr_idx, curr_idx + c_len, chain_roles.get(c_id, ""))
        curr_idx += c_len

    vh_chains = [k for k, v in chain_roles.items() if v == 'VH']
    vl_chains = [k for k, v in chain_roles.items() if v == 'VL']
    vhh_chains = [k for k, v in chain_roles.items() if v == 'VHH']

    paired_fv_units = [
        (vh_chains[i], vl_chains[i]) for i in range(min(len(vh_chains), len(vl_chains)))
    ]
    all_antibody_units = list(paired_fv_units)
    for extra_vh in vh_chains[len(vl_chains):]:
        all_antibody_units.append((extra_vh, None))
    for vhh_c in vhh_chains:
        all_antibody_units.append((vhh_c, None))

    has_paired_fv = len(paired_fv_units) > 0
    has_any_ab_domain = len(all_antibody_units) > 0

    def slice_chain(full_seq_str: str, c_id: str) -> str:
        s_idx, e_idx, _ = chain_bounds[c_id]
        return full_seq_str[s_idx:e_idx]

    # Helper to map a global 0-based residue index to (chain_id, local_0idx, role)
    def map_global_to_chain(global_0idx: int):
        for c_id, (s_idx, e_idx, role) in chain_bounds.items():
            if s_idx <= global_0idx < e_idx:
                return c_id, global_0idx - s_idx, role
        return None, None, ""

    # ---------------------------------------------------------
    # 1. Parse Sequences & Extract Primary Sub-domains (VH/VL)
    # ---------------------------------------------------------
    variants = []
    for i, record in enumerate(records):
        unique_id = f"{run_name}_WT" if i == 0 else record.id
        seq = str(record.seq).replace(":", "").replace("/", "")
        
        # Store primary VH/VHH and VL for table display (multi-arm scoring uses slice_chain directly)
        primary_vh_id = vh_chains[0] if vh_chains else (vhh_chains[0] if vhh_chains else None)
        primary_vl_id = vl_chains[0] if vl_chains else None
        vh_seq = slice_chain(seq, primary_vh_id) if primary_vh_id else None
        vl_seq = slice_chain(seq, primary_vl_id) if primary_vl_id else None
        
        # Calculate mutations relative to WT
        mutations = []
        if seq == clean_wt:
            mut_string = "WT"
        else:
            for idx, (wt_aa, mut_aa) in enumerate(zip(clean_wt, seq)):
                if wt_aa != mut_aa:
                    mutations.append(f"{wt_aa}{idx+1}{mut_aa}")
            mut_string = ",".join(mutations) if mutations else "WT"
            
        variants.append({
            "id": unique_id, 
            "mutations": mut_string, 
            "full_sequence": seq,
            "VH": vh_seq,
            "VL": vl_seq
        })

    df = pd.DataFrame(variants)

    # ---------------------------------------------------------
    # 2. METRIC-LEVEL SMART CACHE MERGE
    # ---------------------------------------------------------
    if os.path.exists(final_csv_path):
        try:
            cached_df = pd.read_csv(final_csv_path)
            # Drop previous normalized/composite columns so they get cleanly recomputed
            cols_to_drop = [
                c for c in cached_df.columns
                if c.endswith('_norm') or c.startswith('z_') or c.startswith('raw_') or c in [
                    'global_composite_score', 'agnostic_gated_score', 'legacy_linear_score',
                    'passed_all_gates', 'gate_status', 'source_model',
                    'pi_delta', 'protein_format', 'num_mutations'
                ]
            ]
            cached_df = cached_df.drop(columns=cols_to_drop, errors='ignore')
            df = df.merge(
                cached_df.drop(columns=['mutations', 'full_sequence', 'VH', 'VL'], errors='ignore'),
                on='id', how='left'
            )
            print(f" -> Cache loaded. Evaluating missing metrics for {len(df)} sequences...")
        except Exception as e:
            print(f" -> Could not verify cache: {e}. Recomputing from scratch...")

    # ---------------------------------------------------------
    # 3. Web Server CamSol Scores (Fail-Fast Verification)
    # ---------------------------------------------------------
    if 'camsol_score' not in df.columns:
        df['camsol_score'] = np.nan

    if os.path.exists(camsol_txt_path):
        try:
            camsol_df = pd.read_csv(camsol_txt_path, sep='\t')
        except Exception as e:
            raise ValueError(f"Failed to read CamSol file at {camsol_txt_path}: {e}")

        score_col = 'protein variant score'
        if score_col not in camsol_df.columns:
            raise ValueError(
                f"Invalid CamSol file: '{score_col}' column not found in '{camsol_txt_path}'. "
                f"Found columns: {list(camsol_df.columns)}"
            )

        # 1. Resolve ID column (CamSol defaults to 'Name')
        id_col = 'Name' if 'Name' in camsol_df.columns else camsol_df.columns[0]
        camsol_ids = set(camsol_df[id_col].astype(str).str.strip().str.lstrip(">"))
        expected_ids = set(df['id'].astype(str))
        matching_ids = expected_ids.intersection(camsol_ids)

        # HARD STOP: Abort immediately if the file belongs to a different campaign
        if len(matching_ids) == 0:
            sample_expected = list(expected_ids)[:3]
            sample_found = list(camsol_ids)[:3]
            raise ValueError(
                f"\n[CRITICAL ERROR] Wrong CamSol file uploaded for '{run_name}'!\n"
                f" -> Expected variant IDs (e.g., {sample_expected})\n"
                f" -> Found in file (e.g., {sample_found})\n"
                f"Execution halted immediately to prevent running PyRosetta on mismatched data."
            )

        # 2. SEQUENCE INTEGRITY CHECK (Ensures same ID doesn't mask different mutations)
        seq_col = next((c for c in camsol_df.columns if 'seq' in c.lower()), None)
        if seq_col:
            camsol_seq_map = dict(zip(
                camsol_df[id_col].astype(str).str.strip().str.lstrip(">"),
                camsol_df[seq_col].astype(str).str.strip().str.replace(":", "").str.replace("/", "")
            ))
            mismatched_variants = []
            for _, row in df[df['id'].isin(matching_ids)].iterrows():
                expected_seq = row['full_sequence']
                actual_camsol_seq = camsol_seq_map.get(row['id'], "")
                if actual_camsol_seq and actual_camsol_seq != expected_seq:
                    mismatched_variants.append(row['id'])

            if mismatched_variants:
                raise ValueError(
                    f"\n[CRITICAL ERROR] CamSol sequence content mismatch for {len(mismatched_variants)} variant(s)!\n"
                    f" -> Affected IDs: {mismatched_variants[:3]}\n"
                    f"The IDs matched, but the amino acid sequences in '{camsol_txt_path}' do not match "
                    f"this campaign's actual sequences. Aborting before PyRosetta calculation."
                )

        # 3. Safe mapping
        score_mapping = dict(zip(
            camsol_df[id_col].astype(str).str.strip().str.lstrip(">"),
            camsol_df[score_col].astype(float)
        ))
        df['camsol_score'] = df['id'].map(score_mapping)
        matched_count = df['camsol_score'].notna().sum()

        if matched_count < len(df):
            print(f" -> WARNING: Partial CamSol match: {matched_count}/{len(df)} variants mapped.")
        else:
            print(f" -> CamSol successfully verified and mapped ({matched_count}/{len(df)} variants).")

    # ---------------------------------------------------------
    # 4. Sequence Biophysics (pI, GRAVY, Net Charge, Multi-Arm Asymmetry)
    # ---------------------------------------------------------
    if 'pI' not in df.columns: df['pI'] = np.nan
    if 'gravy' not in df.columns: df['gravy'] = np.nan
    if 'net_charge' not in df.columns: df['net_charge'] = np.nan
    if 'charge_asymmetry' not in df.columns: df['charge_asymmetry'] = np.nan
    
    mask = df['pI'].isna() | df['gravy'].isna() | df['net_charge'].isna() | (has_paired_fv & df['charge_asymmetry'].isna())
    if mask.any():
        print(f"Calculating sequence biophysics for {mask.sum()} variants...")
        for idx, row in df[mask].iterrows():
            full_seq_str = row['full_sequence']
            full_analysis = ProteinAnalysis(full_seq_str)
            df.at[idx, 'net_charge'] = full_analysis.charge_at_pH(7.4)
            df.at[idx, 'pI'] = full_analysis.isoelectric_point()
            df.at[idx, 'gravy'] = full_analysis.gravy()
            
            # Compute mean VH-VL dipole across all paired Fv arms (supports both 2-chain Fv and 4-chain Bispecifics)
            if has_paired_fv:
                arm_dipoles = []
                for h_id, l_id in paired_fv_units:
                    h_s = slice_chain(full_seq_str, h_id)
                    l_s = slice_chain(full_seq_str, l_id)
                    if h_s and l_s:
                        q_h = ProteinAnalysis(h_s).charge_at_pH(7.4)
                        q_l = ProteinAnalysis(l_s).charge_at_pH(7.4)
                        arm_dipoles.append(abs(q_h - q_l))
                if arm_dipoles:
                    df.at[idx, 'charge_asymmetry'] = float(np.mean(arm_dipoles))
    else:
        print(" -> Sequence biophysics up to date.")
        
    df['pi_delta'] = (df['pI'] - 7.4).abs()

    # ---------------------------------------------------------
    # 5. PyRosetta Structural Stability (ddG)
    # ---------------------------------------------------------
    if 'pyrosetta_ddg' not in df.columns: df['pyrosetta_ddg'] = np.nan
    mask = df['pyrosetta_ddg'].isna()
    
    if mask.any():
        print(f"Calculating PyRosetta for {mask.sum()} missing variants...")
        relaxed_pdb_path = os.path.join(run_dir, f"{run_name}_relaxed.pdb")
        
        if os.path.exists(relaxed_pdb_path):
            safe_init_pyrosetta()  
            import pyrosetta
            from pyrosetta.toolbox import mutate_residue 

            wt_pose = pyrosetta.pose_from_file(relaxed_pdb_path)
            scorefxn = pyrosetta.get_fa_scorefxn()
            wt_energy = scorefxn(wt_pose)
            pose_length = wt_pose.total_residue()
            
            for idx, row in df[mask].iterrows():
                if row['mutations'] == "WT":
                    df.at[idx, 'pyrosetta_ddg'] = 0.0
                    continue

                mut_pose = wt_pose.clone()
                try:
                    for m in row['mutations'].split(","):
                        m = m.strip()
                        if not m: continue
                        pos, mut_aa = int(m[1:-1]), m[-1]
                        
                        if 1 <= pos <= pose_length:
                            # Safeguard prevents C++ corruption
                            if mut_pose.residue(pos).name3() == "CYS" and mut_pose.residue(pos).has_variant_type(pyrosetta.rosetta.core.chemical.DISULFIDE):
                                print(f" -> Skipping {m}: Protects C++ core from disulfide segfault.")
                                continue
                            mutate_residue(mut_pose, pos, mut_aa, pack_scorefxn=scorefxn)
                        else:
                            print(f" -> Skipping {m}: Out of bounds")

                    df.at[idx, 'pyrosetta_ddg'] = scorefxn(mut_pose) - wt_energy
                except Exception as e:
                    print(f" -> PyRosetta warning for {row['mutations'][:30]}... : {e}")
                    df.at[idx, 'pyrosetta_ddg'] = 0.0
                finally:
                    # Now safe to clear because the C++ object is never corrupted!
                    mut_pose.clear()
                    del mut_pose
                    
            # Clear the WT pose and flush the RAM
            wt_pose.clear()
            del wt_pose
            gc.collect() 
        else:
            df['pyrosetta_ddg'] = 0.0
    else:
        print(" -> PyRosetta metrics up to date. Skipping model.")
        
    # ---------------------------------------------------------
    # 6. PLM 1: AntiBERTy (Per-Chain Antibody Fitness — Safe for 4-Chain Bispecifics)
    # ---------------------------------------------------------
    if has_any_ab_domain:
        if 'antiberty_fitness' not in df.columns: df['antiberty_fitness'] = np.nan
        mask = df['antiberty_fitness'].isna()
        
        if mask.any():
            print(f"Loading AntiBERTy to score {mask.sum()} missing antibody variants per domain...")
            ab_tokenizer = AutoTokenizer.from_pretrained("jeffruffolo/AntiBERTy")
            ab_model = AutoModelForMaskedLM.from_pretrained("jeffruffolo/AntiBERTy").to(device)
            ab_model.eval()

            wt_chain_lists = {c_id: list(slice_chain(clean_wt, c_id)) for c_id in chain_lengths}

            for idx, row in df[mask].iterrows():
                if row['mutations'] == "WT":
                    df.at[idx, 'antiberty_fitness'] = 0.0
                    continue
                    
                mut_delta_logp = 0.0
                for m in row['mutations'].split(","):
                    m = m.strip()
                    if not m: continue
                    try:
                        wt_aa, global_pos_0idx, mut_aa = m[0], int(m[1:-1]) - 1, m[-1]
                        c_id, local_pos, role = map_global_to_chain(global_pos_0idx)
                        if c_id is None or role not in ["VH", "VL", "VHH"]:
                            continue
                        masked_chain = wt_chain_lists[c_id].copy()
                        if local_pos >= len(masked_chain) or len(masked_chain) > 508:
                            continue
                        masked_chain[local_pos] = "[MASK]"
                        
                        inputs = ab_tokenizer(" ".join(masked_chain), return_tensors="pt").to(device)
                        with torch.no_grad():
                            logits = ab_model(**inputs).logits
                            log_probs = torch.log_softmax(logits[0, local_pos + 1], dim=-1)
                            mut_delta_logp += (
                                log_probs[ab_tokenizer.convert_tokens_to_ids(mut_aa)] - 
                                log_probs[ab_tokenizer.convert_tokens_to_ids(wt_aa)]
                            ).item()
                    except Exception as ex:
                        print(f" -> AntiBERTy warning on {m}: {ex}")
                df.at[idx, 'antiberty_fitness'] = mut_delta_logp
                
            del ab_model, ab_tokenizer
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print(" -> AntiBERTy metrics up to date.")

    # ---------------------------------------------------------
    # 7. PLM 2: AbLang2 (Supports Paired Fv, 4-Chain Bispecifics, and Single-Domain VHH)
    # ---------------------------------------------------------
    if has_any_ab_domain:
        if 'ablang_fitness' not in df.columns: df['ablang_fitness'] = np.nan
        mask = df['ablang_fitness'].isna()
        
        if mask.any():
            print(f"Loading AbLang2 to score {mask.sum()} missing variants across {len(all_antibody_units)} antibody unit(s)...")
            import ablang2
            ablang_model = ablang2.pretrained("ablang2-paired", random_init=False) 
            
            # Precompute WT baseline score for each antibody unit
            wt_unit_scores = []
            for h_id, l_id in all_antibody_units:
                wt_h = slice_chain(clean_wt, h_id)
                wt_l = slice_chain(clean_wt, l_id) if l_id is not None else ""
                wt_unit_scores.append(float(ablang_model([[wt_h, wt_l]], mode='pseudo_log_likelihood')[0]))
            
            for idx, row in df[mask].iterrows():
                if row['mutations'] == "WT":
                    df.at[idx, 'ablang_fitness'] = 0.0
                    continue
                    
                full_s = row['full_sequence']
                total_ablang_delta = 0.0
                for unit_idx, (h_id, l_id) in enumerate(all_antibody_units):
                    mut_h = slice_chain(full_s, h_id)
                    mut_l = slice_chain(full_s, l_id) if l_id is not None else ""
                    mut_s = float(ablang_model([[mut_h, mut_l]], mode='pseudo_log_likelihood')[0])
                    total_ablang_delta += (mut_s - wt_unit_scores[unit_idx])
                    
                df.at[idx, 'ablang_fitness'] = total_ablang_delta

            del ablang_model
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print(" -> AbLang2 metrics up to date.")

    # ---------------------------------------------------------
    # 8. PLM 3: ProtBERT (Evolutionary Fitness)
    # ---------------------------------------------------------
    if 'protbert_fitness' not in df.columns: df['protbert_fitness'] = np.nan
    mask = df['protbert_fitness'].isna()
    
    if mask.any():
        print(f"Loading ProtBERT to score {mask.sum()} missing variants...")
        pb_tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
        pb_model = AutoModelForMaskedLM.from_pretrained("Rostlab/prot_bert").to(device)
        pb_model.eval()

        for idx, row in df[mask].iterrows():
            if row['mutations'] == "WT":
                df.at[idx, 'protbert_fitness'] = 0.0
                continue
                
            mut_delta_logp = 0.0
            for m in row['mutations'].split(","):
                m = m.strip()
                if not m: continue
                try:
                    wt_aa, pos, mut_aa = m[0], int(m[1:-1]) - 1, m[-1]
                    if pos >= len(wt_list): continue
                    masked_seq = wt_list.copy()
                    masked_seq[pos] = "[MASK]" 
                    
                    inputs = pb_tokenizer(" ".join(masked_seq), return_tensors="pt").to(device)
                    with torch.no_grad():
                        logits = pb_model(**inputs).logits
                        log_probs = torch.log_softmax(logits[0, pos + 1], dim=-1)
                        mut_delta_logp += (
                            log_probs[pb_tokenizer.convert_tokens_to_ids(mut_aa)] - 
                            log_probs[pb_tokenizer.convert_tokens_to_ids(wt_aa)]
                        ).item()
                except Exception: pass
            df.at[idx, 'protbert_fitness'] = mut_delta_logp

        del pb_model, pb_tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(" -> ProtBERT metrics up to date.")

    # ---------------------------------------------------------
    # 9. PLM 4: ESM-2 (General Fitness) & Multi-Arm BINDPRED
    # ---------------------------------------------------------
    if 'esm_fitness' not in df.columns: df['esm_fitness'] = np.nan
    if 'bindpred_kd' not in df.columns: df['bindpred_kd'] = np.nan
    
    # Separate masks so VHHs/Enzymes (where bindpred_kd is NaN) don't invalidate the ESM-2 cache!
    mask_esm = df['esm_fitness'].isna()
    mask_bp = df['bindpred_kd'].isna() if has_paired_fv else pd.Series(False, index=df.index)
    mask = mask_esm | mask_bp
    
    if mask.any():
        print(f"Loading ESM-2 & BINDPRED to score {mask.sum()} missing variants...")
        import esm
        from huggingface_hub import hf_hub_download
        from catboost import CatBoostRegressor
        
        model_esm, alphabet_esm = esm.pretrained.esm2_t33_650M_UR50D()
        model_esm = model_esm.to(device)
        batch_converter_esm = alphabet_esm.get_batch_converter()
        model_esm.eval()

        bindpred_model = None
        if mask_bp.any():
            bindpred_model = CatBoostRegressor()
            bindpred_model.load_model(
                hf_hub_download(repo_id="hbp5181/BindPred", filename="ESM2_BindPred.cbm"),
                format="cbm"
            )

        for idx, row in df[mask].iterrows():
            full_s = row['full_sequence']
            # 1. Multi-Arm BINDPRED (Evaluates all paired VH-VL arms in Fvs and Bispecifics)
            if mask_bp.loc[idx] and bindpred_model is not None:
                arm_kds = []
                for h_id, l_id in paired_fv_units:
                    h_s = slice_chain(full_s, h_id)
                    l_s = slice_chain(full_s, l_id)
                    if h_s and l_s:
                        _, _, batch_tokens = batch_converter_esm([("VH", h_s), ("VL", l_s)])
                        with torch.no_grad():
                            reps = model_esm(batch_tokens.to(device), repr_layers=[33], return_contacts=False)["representations"][33]
                            pred_kd = bindpred_model.predict(np.hstack([
                                reps[0, 1:len(h_s)+1].mean(0).cpu().numpy(),
                                reps[1, 1:len(l_s)+1].mean(0).cpu().numpy()
                            ]))
                            arm_kds.append(float(pred_kd))
                if arm_kds:
                    df.at[idx, 'bindpred_kd'] = float(np.mean(arm_kds))

            # 2. ESM-2 Masked-Marginal Log-Likelihood
            if mask_esm.loc[idx]:
                if row['mutations'] == "WT":
                    df.at[idx, 'esm_fitness'] = 0.0
                else:
                    mut_delta_logp = 0.0
                    for m in row['mutations'].split(","):
                        m = m.strip()
                        if not m: continue
                        try:
                            wt_aa, pos, mut_aa = m[0], int(m[1:-1]) - 1, m[-1]
                            if pos >= len(wt_list): continue
                            masked_seq = wt_list.copy()
                            masked_seq[pos] = "<mask>"
                            
                            _, _, mask_tokens = batch_converter_esm([("seq", "".join(masked_seq))])
                            with torch.no_grad():
                                log_probs = torch.log_softmax(model_esm(mask_tokens.to(device))["logits"][0, pos + 1], dim=-1)
                                mut_delta_logp += (
                                    log_probs[alphabet_esm.tok_to_idx[mut_aa]] -
                                    log_probs[alphabet_esm.tok_to_idx[wt_aa]]
                                ).item()
                        except Exception: pass
                    df.at[idx, 'esm_fitness'] = mut_delta_logp
                
        del model_esm, bindpred_model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(" -> ESM-2 & BINDPRED metrics up to date.")

    # ---------------------------------------------------------
    # 10. PLM 5: POET-2 (OpenProtein.AI API — Disabled by Default)
    # ---------------------------------------------------------
    if 'poet2_fitness' not in df.columns: df['poet2_fitness'] = np.nan
    mask = df['poet2_fitness'].isna()
    
    if False:  # Retained exactly as requested
        print(f"Calling OpenProtein API (POET-2) to score {mask.sum()} missing variants...")
        
        try:
            import openprotein
            
            # 1. Authenticate
            op_user = os.environ.get("OPENPROTEIN_USERNAME")
            op_pass = os.environ.get("OPENPROTEIN_PASSWORD")
            session = openprotein.connect(username=op_user, password=op_pass)
            
            # Gather WT and variants (WT must be first for relative scoring)
            seqs = [clean_wt]
            index_mapping = []
            
            for idx, row in df[mask].iterrows():
                if row['mutations'] != "WT":
                    seqs.append(row['full_sequence'])
                    index_mapping.append(idx)
                    
            if len(seqs) > 1:
                # 2. Create MSA exactly as the tutorial outlines
                print(" -> Creating MSA...")
                msa = session.align.create_msa(clean_wt.encode())
                msa.wait()
                
                # 3. Sample Prompt
                print(" -> Generating Evolutionary Prompt...")
                prompt = msa.sample_prompt(num_ensemble_prompts=1, random_seed=42)
                prompt.wait()
                
                # 4. Score Sequences
                print(f" -> Submitting PoET scoring job for {len(seqs)} sequences...")
                poet = session.embedding.get_model("poet")
                scorejob = poet.score(prompt=prompt.prompt_id, sequences=seqs)
                score_results = scorejob.wait()
                
                # 5. Process Results
                # NB: one score per prompt. We used 1 ensemble prompt, so results are in index 0.
                scores = score_results[0]
                
                # Helper to safely extract the numeric score regardless of API return type
                def get_val(s):
                    if hasattr(s, 'score'): return s.score
                    if hasattr(s, 'log_likelihood'): return s.log_likelihood
                    if isinstance(s, dict) and 'log_likelihood' in s: return s['log_likelihood']
                    return float(s)
                
                wt_score = get_val(scores[0])
                
                for idx, mut_score_obj in zip(index_mapping, scores[1:]):
                    df.at[idx, 'poet2_fitness'] = get_val(mut_score_obj) - wt_score
                    
                if "WT" in df.loc[mask, 'mutations'].values:
                    df.loc[mask & (df['mutations'] == "WT"), 'poet2_fitness'] = 0.0
                    
                print(" -> POET API scoring complete.")
            else:
                df.loc[mask & (df['mutations'] == "WT"), 'poet2_fitness'] = 0.0
                
        except Exception as e:
            print(f" -> WARNING: OpenProtein API call failed: {e}")
    else:
        print(" -> POET metrics up to date. Skipping API call.")
    
    # ---------------------------------------------------------
    # 11. PLM 6: ESM-C (Biohub 300M via Isolated Env)
    # ---------------------------------------------------------
    if 'esmc_fitness' not in df.columns: df['esmc_fitness'] = np.nan
    mask = df['esmc_fitness'].isna()
    
    if mask.any():
        print(f"Running ESM-C in isolated env to score {mask.sum()} missing variants...")
        temp_in_csv = os.path.join(run_dir, "temp_esmc_in.csv")
        temp_out_csv = os.path.join(run_dir, "temp_esmc_out.csv")
        
        df.loc[mask, ['id', 'mutations']].to_csv(temp_in_csv, index=False)
        
        esmc_score_cmd = [
            "conda", "run", "-n", "esmc_env",
            "python", "run_esmc_scoring.py",
            "--wt_seq", clean_wt,
            "--input_csv", temp_in_csv,
            "--output_csv", temp_out_csv
        ]
        try:
            subprocess.run(esmc_score_cmd, check=True, capture_output=True, text=True)
            if os.path.exists(temp_out_csv):
                esmc_res = pd.read_csv(temp_out_csv)
                esmc_map = dict(zip(esmc_res['id'], esmc_res['esmc_fitness']))
                df.loc[mask, 'esmc_fitness'] = df.loc[mask, 'id'].map(esmc_map)
                os.remove(temp_out_csv)
            if os.path.exists(temp_in_csv):
                os.remove(temp_in_csv)
            print(" -> ESM-C scoring complete.")
        except subprocess.CalledProcessError as e:
            print(f" -> WARNING: ESM-C scoring failed:\n{e.stderr}")
    else:
        print(" -> ESM-C metrics up to date.")

    # ---------------------------------------------------------
    # 12. JanusDDG Scoring
    # ---------------------------------------------------------
    if 'janus_ddg' not in df.columns: df['janus_ddg'] = np.nan
    mask = df['janus_ddg'].isna()
    
    if mask.any():
        print(f"Running JanusDDG for {mask.sum()} missing variants...")
        janus_src_path = "/home/esigalas/workspace/JanusDDG/src" 
        if janus_src_path not in sys.path:
            sys.path.append(janus_src_path)
        try:
            import utils
            utils.MODELS_DIR = "/home/esigalas/workspace/JanusDDG/models" 
            
            temp_csv_path = os.path.join(run_dir, "temp_janus_input.csv")
            subset_to_compute = df[mask & (df['mutations'] != "WT")]
            
            if len(subset_to_compute) > 0:
                pd.DataFrame({
                    'ID': subset_to_compute['id'], 
                    'Sequence': clean_wt, 
                    'MTS': subset_to_compute['mutations'].str.replace(",", "_")
                }).to_csv(temp_csv_path, index=False)
                
                import __main__
                import model as janus_model
                import esm
                
                model_esm, alphabet_esm = esm.pretrained.esm2_t33_650M_UR50D()
                model_esm = model_esm.to(device)
                batch_converter_esm = alphabet_esm.get_batch_converter()
                model_esm.eval()
                
                janus_model.device = device
                setattr(__main__, 'Cross_Attention_DDG', janus_model.Cross_Attention_DDG)
                setattr(__main__, 'TransformerRegression', janus_model.TransformerRegression)
                setattr(__main__, 'SinusoidalPositionalEncoding', janus_model.SinusoidalPositionalEncoding)

                model_janus = utils.load_model('JanusDDG_fine_tuned.pth', device)
                pred_dir, _ = utils.process_and_predict(temp_csv_path, model_janus, model_esm, batch_converter_esm, device)
                
                janus_mapping = dict(zip(subset_to_compute['id'], pred_dir.values))
                df.loc[mask, 'janus_ddg'] = df.loc[mask, 'id'].map(janus_mapping).fillna(0.0)
                os.remove(temp_csv_path)
                
                del model_esm, model_janus
                gc.collect()
                torch.cuda.empty_cache()
            else:
                df.loc[mask, 'janus_ddg'] = 0.0
        except Exception as e:
            print(f" -> WARNING: JanusDDG failed: {e}")
    else:
        print(" -> JanusDDG metrics up to date.")

    # ---------------------------------------------------------
    # Safely Initialize Missing Columns to Prevent UI Plot Crashes
    # ---------------------------------------------------------
    expected_cols = [
        'janus_ddg', 'esm_fitness', 'esmc_fitness', 'antiberty_fitness', 
        'ablang_fitness', 'protbert_fitness', 'poet2_fitness', 'bindpred_kd', 'charge_asymmetry'
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    # ---------------------------------------------------------
    # 13. Legacy Global Composite Score AND Agnostic Gated Score
    # ---------------------------------------------------------
    def norm(col_or_series, invert=False):
        s = df[col_or_series] if isinstance(col_or_series, str) else col_or_series
        if s is None or s.isna().all():
            return pd.Series(np.nan, index=df.index)
        c_min, c_max = s.min(), s.max()
        if c_max == c_min:
            return pd.Series(0.5, index=df.index)
        return (c_max - s) / (c_max - c_min) if invert else (s - c_min) / (c_max - c_min)

    def zscore_series(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        if len(valid) <= 1 or valid.nunique() <= 1:
            return pd.Series(0.0, index=s.index)
        std = valid.std(ddof=0)
        return ((s - valid.mean()) / (std if std > 1e-9 else 1.0)).fillna(0.0)

    df['protein_format'] = protein_format
    df['num_mutations'] = df['mutations'].apply(
        lambda m: 0 if str(m).strip() in ['WT', 'nan', ''] else len(str(m).split(','))
    )
    df['source_model'] = df['id'].apply(
        lambda s: "WT" if str(s).endswith("_WT") else (
            "Consensus (" + str(s).split("_")[0] + "+)" if "+" in str(s) else str(s).split("_")[0]
        )
    )

    # --- A. Min-Max Normalized Pillars (0..1) ---
    df['camsol_norm']      = norm(df.get('camsol_score'), invert=False)
    df['janus_norm']       = norm(df.get('janus_ddg'), invert=False)
    df['pi_norm']          = norm(df.get('pi_delta'), invert=False)
    df['charge_asym_norm'] = norm(df.get('charge_asymmetry'), invert=True)
    df['net_charge_norm']  = norm(df.get('net_charge'), invert=False)
    df['gravy_norm']       = norm(df.get('gravy'), invert=True)
    df['bindpred_norm']    = norm(df.get('bindpred_kd'), invert=True)
    df['pyrosetta_norm']   = norm(df.get('pyrosetta_ddg'), invert=True)
    df['esm_norm']         = norm(df.get('esm_fitness'), invert=False)
    df['esmc_norm']        = norm(df.get('esmc_fitness'), invert=False)
    df['ab_norm']          = norm(df.get('antiberty_fitness'), invert=False)
    df['ablang_norm']      = norm(df.get('ablang_fitness'), invert=True)
    df['protbert_norm']    = norm(df.get('protbert_fitness'), invert=False)

    # --- B. Legacy Format-Specific Weighted Composite Score (0..1) ---
    active_weights = FORMAT_WEIGHTS.get(protein_format, FORMAT_WEIGHTS["scFv (Single-chain Fv)"])

    # 4. Compute weighted average across all available (non-NaN) metrics for this format
    weighted_sum = pd.Series(0.0, index=df.index)
    total_weight = 0.0

    for norm_col, weight in active_weights.items():
        if norm_col in df.columns and df[norm_col].notna().any():
            weighted_sum += weight * df[norm_col].fillna(0.5)
            total_weight += weight

    df['global_composite_score'] = weighted_sum / max(total_weight, 1e-6)

    # --- C. Agnostic Gated + Invariant Score (Zero-Shot Z-Score Architecture) ---
    wt_rows = df[df['mutations'] == "WT"]
    wt_ref = wt_rows.iloc[0] if not wt_rows.empty else df.iloc[0]

    def get_delta_vs_wt(col_name: str) -> pd.Series:
        if col_name not in df.columns or df[col_name].isna().all():
            return pd.Series(0.0, index=df.index)
        wt_val = wt_ref[col_name] if pd.notna(wt_ref.get(col_name, np.nan)) else 0.0
        return (df[col_name] - float(wt_val)).fillna(0.0)

    raw_pyrosetta  = get_delta_vs_wt('pyrosetta_ddg')
    raw_bindpred   = get_delta_vs_wt('bindpred_kd')
    raw_dipole     = get_delta_vs_wt('charge_asymmetry')
    raw_pi_delta   = get_delta_vs_wt('pi_delta')
    raw_camsol     = get_delta_vs_wt('camsol_score')
    raw_esm        = get_delta_vs_wt('esm_fitness')
    raw_janus      = get_delta_vs_wt('janus_ddg')

    # Within-campaign Z-scores for the Invariant Core & Downside Folding Hinges
    z_pyrosetta    = zscore_series(raw_pyrosetta)
    z_bindpred     = zscore_series(raw_bindpred)
    z_dipole       = zscore_series(raw_dipole)
    z_pi_delta     = zscore_series(raw_pi_delta)
    z_num_muts     = zscore_series(df['num_mutations'].astype(float))

    z_hinge_esm    = zscore_series(np.minimum(0.0, raw_esm))
    z_hinge_janus  = zscore_series(np.minimum(0.0, raw_janus))
    z_hinge_camsol = zscore_series(np.minimum(0.0, raw_camsol))

    # Non-Compensatory Biophysical Gates
    clash_fail     = ((z_pyrosetta > 0.67) & (raw_pyrosetta > 0.0)) | (raw_pyrosetta > 5.0)
    interface_fail = has_paired_fv & (z_bindpred > 0.67) & (raw_bindpred > 0.0)
    unfold_fail    = (raw_esm < -10.0) | (raw_janus < -0.5)

    df['passed_all_gates'] = ~(clash_fail | interface_fail | unfold_fail)
    df['gate_status'] = [
        "PASS" if p else "FAIL (" + "+".join(
            (["3D_Clash"] if c else []) +
            (["Interface_Kd"] if i else []) +
            (["1D_Unfolding"] if u else [])
        ) + ")"
        for p, c, i, u in zip(df['passed_all_gates'], clash_fail, interface_fail, unfold_fail)
    ]

    invariant_core = (
        -0.30 * z_pyrosetta +
        -0.25 * z_bindpred +
        -0.20 * z_dipole +
        +0.15 * z_pi_delta +
        -0.10 * z_num_muts
    )
    downside_hinge = (
        +0.25 * z_hinge_esm +
        +0.15 * z_hinge_janus +
        +0.10 * z_hinge_camsol
    )
    gate_penalty = -0.50 * (
        clash_fail.astype(float) + interface_fail.astype(float) + unfold_fail.astype(float)
    )

    df['agnostic_gated_score'] = invariant_core + downside_hinge + gate_penalty
    df['agnostic_gated_norm']  = norm(df['agnostic_gated_score'], invert=False)

    # Sort primarily by gate pass + agnostic_gated_score (both scores remain side-by-side in the CSV)
    df = df.sort_values(
        by=['passed_all_gates', 'agnostic_gated_score'],
        ascending=[False, False]
    ).reset_index(drop=True)
    
    # 5. Clean up columns for the final CSV layout
    ordered_cols = [
        'id', 'source_model', 'mutations', 'protein_format', 'num_mutations',
        'agnostic_gated_score', 'global_composite_score', 'passed_all_gates', 'gate_status',
        'janus_ddg', 'pyrosetta_ddg', 'camsol_score', 'pI', 'pi_delta', 'gravy', 'net_charge',
        'charge_asymmetry', 'bindpred_kd', 'esm_fitness', 'antiberty_fitness', 
        'ablang_fitness', 'protbert_fitness', 'poet2_fitness', 'esmc_fitness', 'full_sequence'
    ]
    cols_to_save = [c for c in ordered_cols if c in df.columns] + [
        c for c in df.columns if c.endswith('_norm') or c in ['VH', 'VL']
    ]
    
    df[cols_to_save].to_csv(final_csv_path, index=False)
    print(f"Scoring complete for format '{protein_format}'. Results saved to {final_csv_path}")
    
    return df
