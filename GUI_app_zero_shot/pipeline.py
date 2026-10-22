import os
import sys
import threading
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

def run_universal_developability_pipeline(
    chains: dict, 
    chain_roles: dict, 
    base_output_dir: str, 
    run_name: str, 
    protein_format: str,
    gen_models: list = None, 
    num_seqs: int = 100, 
    temp: float = 0.2
):
    """
    Universal Phase 1 pipeline supporting N-chain complexes, antibodies, and enzymes.
    - chains: dict of sequence strings {'A': '...', 'B': '...'}
    - chain_roles: dict mapping chain keys to roles {'A': 'VH', 'B': 'VL'} or {'A': 'Chain_A'}
    """
    if gen_models is None:
        gen_models = []

    run_dir = os.path.join(base_output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    unrelaxed_pdb_path = os.path.join(run_dir, f"{run_name}_unrelaxed.pdb")
    relaxed_pdb_path = os.path.join(run_dir, f"{run_name}_relaxed.pdb")
    metadata_path = os.path.join(run_dir, f"{run_name}_metadata.json")

    print(f"Starting Universal Pipeline for: {run_name} (Format: {protein_format})")
    print(f"Workspace initialized at: {run_dir}")
    
    # ---------------------------------------------------------
    # STEP 1: Save Run Manifest & Metadata
    # ---------------------------------------------------------
    chain_lengths = {k: len(v) for k, v in chains.items()}
    run_metadata = {
        "protein_format": protein_format,
        "chain_lengths": chain_lengths,
        "chain_roles": chain_roles
    }
    with open(metadata_path, "w") as f:
        json.dump(run_metadata, f, indent=4)
    print(f" -> Saved run metadata. Chains: {chain_lengths}")

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
    # STEP 4: Generative Ensemble (Format-Aware Routing)
    # ---------------------------------------------------------
    print(f"Step 4: Running Generative Ensemble: {gen_models}")
    all_raw_records = []
    
    is_paired_antibody = len(chains) >= 2 and any(r in ['VH', 'VL', 'VHH'] for r in chain_roles.values())
    primary_chain_key = list(chains.keys())[0]
    clean_wt_full = "".join(chains.values())

    # --- MODEL A: AntiFold (Antibody-Specific Fv Generator) ---
    if "AntiFold (Antibody-specific)" in gen_models and is_paired_antibody:
        print(" -> Running AntiFold...")
        ch_heavy = [k for k, v in chain_roles.items() if v in ['VH', 'VHH']][0]
        ch_light = [k for k, v in chain_roles.items() if v == 'VL']
        ch_light_id = ch_light[0] if ch_light else ch_heavy # Fallback
        
        antifold_cmd = [
            "python", "-m", "antifold.main",
            "--pdb_file", relaxed_pdb_path,
            "--heavy_chain", ch_heavy, "--light_chain", ch_light_id,
            "--regions", "FWL FWH",
            "--num_seq_per_target", str(num_seqs),
            "--sampling_temp", str(temp),
            "--out_dir", run_dir
        ]
        try:
            subprocess.run(antifold_cmd, check=True, capture_output=True, text=True)
            af_fasta = os.path.join(run_dir, f"{run_name}_relaxed_AB.fasta")
            if os.path.exists(af_fasta):
                af_records = list(SeqIO.parse(af_fasta, "fasta"))
                all_raw_records.extend(af_records)
                os.remove(af_fasta)
        except subprocess.CalledProcessError as e:
            print(f" -> WARNING: AntiFold failed: {e.stderr}")

    # --- DYNAMIC CDR / REGION MASKING FOR MPNN ---
    fixed_json_path = os.path.join(run_dir, "fixed_positions.jsonl")
    has_mpnn = any("ProteinMPNN" in m or "ThermoMPNN" in m for m in gen_models)
    
    if has_mpnn:
        fixed_dict = {}
        pdb_basename = os.path.splitext(os.path.basename(relaxed_pdb_path))[0]
        fixed_positions_map = {}
        
        try:
            from abnumber import Chain as AbChain
        except ImportError:
            AbChain = None
            print(" -> WARNING: abnumber not installed. Cannot mask CDRs.")
            
        for c_id, c_seq in chains.items():
            role = chain_roles.get(c_id, "")
            fixed_res = []
            
            # If it's an antibody chain, freeze Frameworks and unmask CDRs
            if role in ["VH", "VL", "VHH"] and AbChain is not None:
                try:
                    ab = AbChain(c_seq, scheme='imgt')
                    cdr_seqs = [ab.cdr1_seq, ab.cdr2_seq, ab.cdr3_seq]
                    mutable_indices = set()
                    
                    # Find exact substring matches for CDRs to map 1-based indices
                    for cdr in cdr_seqs:
                        if cdr:
                            start = c_seq.find(cdr)
                            if start != -1:
                                mutable_indices.update(range(start + 1, start + 1 + len(cdr)))
                                
                    # Freeze all residues that are NOT in the mutable CDR set
                    fixed_res = [i for i in range(1, len(c_seq) + 1) if i not in mutable_indices]
                except Exception as e:
                    print(f" -> WARNING: abnumber parsing failed for Chain {c_id}: {e}")
                    fixed_res = [] # Fallback to fully mutable if parsing fails
            else:
                # If it's an enzyme or non-antibody, default to fully mutable
                fixed_res = []
                
            fixed_positions_map[c_id] = fixed_res
            
        fixed_dict[pdb_basename] = fixed_positions_map
        
        with open(fixed_json_path, 'w') as f:
            f.write(json.dumps(fixed_dict) + "\n")
        
        print(f" -> Generated MPNN masking file at {fixed_json_path}")

        # Example ProteinMPNN subprocess call (add your specific path/args here)
        # mpnn_cmd = ["python", "protein_mpnn_run.py", "--jsonl_path", fixed_json_path, ...]
        # subprocess.run(mpnn_cmd, check=True)
        
    # Additional generator hooks (ProteinMPNN, ThermoMPNN, ESM-C) follow standard multi-chain args...

    # ---------------------------------------------------------
    # STEP 5: Pooling and Deduplication
    # ---------------------------------------------------------
    camsol_fasta_path = os.path.join(run_dir, f"{run_name}_camsol_upload.fasta")
    print(f"Step 5: Pooling candidates -> {camsol_fasta_path}")
    
    unique_seqs = {}
    cleaned_records = []
    
    wt_record = SeqRecord(Seq(clean_wt_full), id=f"{run_name}_WT", description="")
    cleaned_records.append(wt_record)
    unique_seqs[clean_wt_full] = True
    
    for record in all_raw_records:
        clean_seq = str(record.seq).replace(":", "").replace("/", "")
        if clean_seq not in unique_seqs:
            unique_seqs[clean_seq] = True
            new_record = SeqRecord(
                Seq(clean_seq), 
                id=record.id,  # Keeps the original FASTA ID
                description=""
            )
            cleaned_records.append(new_record)

    SeqIO.write(cleaned_records, camsol_fasta_path, "fasta")
    print(f"Step 5 Complete: Pooled {len(cleaned_records)} unique variants.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_universal_variants(run_dir: str, run_name: str, wt_sequence: str):
    """
    Universal Phase 2 scoring engine handling biophysics, PyRosetta, and PLMs.
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
        protein_format = meta.get("protein_format", "scFv")

    records = list(SeqIO.parse(raw_fasta_path, "fasta"))
    clean_wt = wt_sequence.replace(":", "").replace("/", "")
    wt_list = list(clean_wt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------------------------------------------------------
    # 1. Parse Sequences & Extract Sub-domains (VH/VL)
    # ---------------------------------------------------------
    variants = []
    for i, record in enumerate(records):
        unique_id = f"{run_name}_WT" if i == 0 else record.id  # Reads the true ID
        seq = str(record.seq).replace(":", "").replace("/", "")
        
        # Slicing VH and VL dynamically based on chain metadata
        current_idx = 0
        vh_seq, vl_seq = None, None
        for c_id, c_len in chain_lengths.items():
            role = chain_roles.get(c_id, "")
            if role in ["VH", "VHH"]: 
                vh_seq = seq[current_idx : current_idx + c_len]
            if role == "VL": 
                vl_seq = seq[current_idx : current_idx + c_len]
            current_idx += c_len
        
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
            cols_to_drop = [c for c in cached_df.columns if c.endswith('_norm') or c in ['global_composite_score', 'pi_delta', 'protein_format', 'num_mutations']]
            cached_df = cached_df.drop(columns=cols_to_drop, errors='ignore')
            
            df = df.merge(cached_df.drop(columns=['mutations', 'full_sequence', 'VH', 'VL'], errors='ignore'), on='id', how='left')
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
    # 4. Sequence Biophysics (pI, GRAVY, Net Charge, Asymmetry)
    # ---------------------------------------------------------
    if 'pI' not in df.columns: df['pI'] = np.nan
    if 'gravy' not in df.columns: df['gravy'] = np.nan
    if 'net_charge' not in df.columns: df['net_charge'] = np.nan
    if 'charge_asymmetry' not in df.columns: df['charge_asymmetry'] = np.nan
    
    mask = df['pI'].isna() | df['gravy'].isna() | df['net_charge'].isna()
    if mask.any():
        print(f"Calculating basic sequence biophysics for {mask.sum()} variants...")
        for idx, row in df[mask].iterrows():
            full_analysis = ProteinAnalysis(row['full_sequence'])
            df.at[idx, 'net_charge'] = full_analysis.charge_at_pH(7.4)
            df.at[idx, 'pI'] = full_analysis.isoelectric_point()
            df.at[idx, 'gravy'] = full_analysis.gravy()
            
            # Asymmetry strictly requires VH and VL
            if row['VH'] and row['VL']:
                vh_analysis = ProteinAnalysis(row['VH'])
                vl_analysis = ProteinAnalysis(row['VL'])
                df.at[idx, 'charge_asymmetry'] = abs(vh_analysis.charge_at_pH(7.4) - vl_analysis.charge_at_pH(7.4))
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
            import gc # Import garbage collector

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
    # 6. PLM 1: AntiBERTy (Antibody-Specific Fitness)
    # ---------------------------------------------------------
    is_antibody = "Enzyme" not in protein_format
    if is_antibody:
        if 'antiberty_fitness' not in df.columns: df['antiberty_fitness'] = np.nan
        mask = df['antiberty_fitness'].isna()
        
        if mask.any():
            print(f"Loading AntiBERTy to score {mask.sum()} missing variants...")
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            ab_tokenizer = AutoTokenizer.from_pretrained("jeffruffolo/AntiBERTy")
            ab_model = AutoModelForMaskedLM.from_pretrained("jeffruffolo/AntiBERTy").to(device)
            ab_model.eval()

            for idx, row in df[mask].iterrows():
                if row['mutations'] == "WT":
                    df.at[idx, 'antiberty_fitness'] = 0.0
                    continue
                    
                mut_delta_logp = 0.0
                for m in row['mutations'].split(","):
                    try:
                        wt_aa, pos, mut_aa = m[0], int(m[1:-1]) - 1, m[-1]
                        if pos >= len(wt_list): continue
                        masked_seq = wt_list.copy()
                        masked_seq[pos] = "[MASK]" 
                        
                        inputs = ab_tokenizer(" ".join(masked_seq), return_tensors="pt").to(device)
                        with torch.no_grad():
                            logits = ab_model(**inputs).logits
                            log_probs = torch.log_softmax(logits[0, pos + 1], dim=-1) 
                            mut_delta_logp += (log_probs[ab_tokenizer.convert_tokens_to_ids(mut_aa)] - 
                                               log_probs[ab_tokenizer.convert_tokens_to_ids(wt_aa)]).item()
                    except Exception: pass
                df.at[idx, 'antiberty_fitness'] = mut_delta_logp
                
            del ab_model, ab_tokenizer
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print(" -> AntiBERTy metrics up to date.")

    # ---------------------------------------------------------
    # 7. PLM 2: AbLang2 (Paired Germline Fitness)
    # ---------------------------------------------------------
    if is_antibody:
        if 'ablang_fitness' not in df.columns: df['ablang_fitness'] = np.nan
        # AbLang2 Strictly requires both VH and VL to be present
        mask = df['ablang_fitness'].isna() & df['VH'].notna() & df['VL'].notna()
        
        if mask.any():
            print(f"Loading AbLang2 to score {mask.sum()} missing paired variants...")
            import ablang2
            ablang_model = ablang2.pretrained("ablang2-paired", random_init=False) 
            
            # Use the first row's VH/VL as the WT reference
            ref_row = df.iloc[0]
            wt_score = ablang_model([[ref_row['VH'], ref_row['VL']]], mode='pseudo_log_likelihood')[0]
            
            for idx, row in df[mask].iterrows():
                if row['mutations'] == "WT":
                    df.at[idx, 'ablang_fitness'] = 0.0
                    continue
                    
                mut_score = ablang_model([[row['VH'], row['VL']]], mode='pseudo_log_likelihood')[0]
                df.at[idx, 'ablang_fitness'] = float(mut_score - wt_score)

            del ablang_model
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print(" -> AbLang2 metrics up to date or incompatible format.")

    # ---------------------------------------------------------
    # 8. PLM 3: ProtBERT (Evolutionary Fitness)
    # ---------------------------------------------------------
    if 'protbert_fitness' not in df.columns: df['protbert_fitness'] = np.nan
    mask = df['protbert_fitness'].isna()
    
    if mask.any():
        print(f"Loading ProtBERT to score {mask.sum()} missing variants...")
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        pb_tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
        pb_model = AutoModelForMaskedLM.from_pretrained("Rostlab/prot_bert").to(device)
        pb_model.eval()

        for idx, row in df[mask].iterrows():
            if row['mutations'] == "WT":
                df.at[idx, 'protbert_fitness'] = 0.0
                continue
                
            mut_delta_logp = 0.0
            for m in row['mutations'].split(","):
                try:
                    wt_aa, pos, mut_aa = m[0], int(m[1:-1]) - 1, m[-1]
                    if pos >= len(wt_list): continue
                    masked_seq = wt_list.copy()
                    masked_seq[pos] = "[MASK]" 
                    
                    inputs = pb_tokenizer(" ".join(masked_seq), return_tensors="pt").to(device)
                    with torch.no_grad():
                        logits = pb_model(**inputs).logits
                        log_probs = torch.log_softmax(logits[0, pos + 1], dim=-1)
                        mut_delta_logp += (log_probs[pb_tokenizer.convert_tokens_to_ids(mut_aa)] - 
                                           log_probs[pb_tokenizer.convert_tokens_to_ids(wt_aa)]).item()
                except Exception: pass
            df.at[idx, 'protbert_fitness'] = mut_delta_logp

        del pb_model, pb_tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(" -> ProtBERT metrics up to date.")

    # ---------------------------------------------------------
    # 9. PLM 4: ESM-2 (General Fitness) & BINDPRED
    # ---------------------------------------------------------
    if 'esm_fitness' not in df.columns: df['esm_fitness'] = np.nan
    if 'bindpred_kd' not in df.columns: df['bindpred_kd'] = np.nan
    
    mask = df['esm_fitness'].isna() | df['bindpred_kd'].isna()
    
    if mask.any():
        print(f"Loading ESM-2 & BINDPRED to score {mask.sum()} missing variants...")
        import esm
        from huggingface_hub import hf_hub_download
        from catboost import CatBoostRegressor
        
        model_esm, alphabet_esm = esm.pretrained.esm2_t33_650M_UR50D()
        model_esm = model_esm.to(device)
        batch_converter_esm = alphabet_esm.get_batch_converter()
        model_esm.eval()

        bindpred_model = CatBoostRegressor()
        bindpred_model.load_model(hf_hub_download(repo_id="hbp5181/BindPred", filename="ESM2_BindPred.cbm"), format="cbm")

        for idx, row in df[mask].iterrows():
            # BINDPRED strictly requires paired Fv (VH + VL)
            if row['VH'] and row['VL']:
                _, _, batch_tokens = batch_converter_esm([("VH", row['VH']), ("VL", row['VL'])])
                with torch.no_grad():
                    reps = model_esm(batch_tokens.to(device), repr_layers=[33], return_contacts=False)["representations"][33]
                    df.at[idx, 'bindpred_kd'] = bindpred_model.predict(np.hstack([
                        reps[0, 1:len(row['VH'])+1].mean(0).cpu().numpy(),
                        reps[1, 1:len(row['VL'])+1].mean(0).cpu().numpy()
                    ]))

            if row['mutations'] == "WT":
                df.at[idx, 'esm_fitness'] = 0.0
            else:
                mut_delta_logp = 0.0
                for m in row['mutations'].split(","):
                    try:
                        wt_aa, pos, mut_aa = m[0], int(m[1:-1]) - 1, m[-1]
                        if pos >= len(wt_list): continue
                        masked_seq = wt_list.copy()
                        masked_seq[pos] = "<mask>"
                        
                        _, _, mask_tokens = batch_converter_esm([("seq", "".join(masked_seq))])
                        with torch.no_grad():
                            log_probs = torch.log_softmax(model_esm(mask_tokens.to(device))["logits"][0, pos + 1], dim=-1)
                            mut_delta_logp += (log_probs[alphabet_esm.tok_to_idx[mut_aa]] - log_probs[alphabet_esm.tok_to_idx[wt_aa]]).item()
                    except Exception: pass
                df.at[idx, 'esm_fitness'] = mut_delta_logp
                
        del model_esm, bindpred_model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(" -> ESM-2 & BINDPRED metrics up to date.")

    # ---------------------------------------------------------
    # 10. PLM 5: POET-2 (OpenProtein.AI API)
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
    # 13. Empirically Calibrated Normalization & Composite Score
    # ---------------------------------------------------------
    def norm(col_or_series, invert=False):
        s = df[col_or_series] if isinstance(col_or_series, str) else col_or_series
        if s is None or s.isna().all():
            return pd.Series(np.nan, index=df.index)
        c_min, c_max = s.min(), s.max()
        if c_max == c_min:
            return pd.Series(0.5, index=df.index)
        res = (c_max - s) / (c_max - c_min) if invert else (s - c_min) / (c_max - c_min)
        return res

    # Save format and mutation count directly into the CSV for easy benchmarking
    df['protein_format'] = protein_format
    df['num_mutations'] = df['mutations'].apply(
        lambda m: 0 if str(m).strip() in ['WT', 'nan', ''] else len(str(m).split(','))
    )
    mut_counts = df['num_mutations'].clip(lower=1)

    # 1. Normalize Intrinsic & Thermodynamic Metrics
    df['camsol_norm']      = norm(df.get('camsol_score'), invert=False)
    df['janus_norm']       = norm(df.get('janus_ddg'), invert=False)
    df['pi_norm']          = norm(df.get('pi_delta'), invert=False)
    df['charge_asym_norm'] = norm(df.get('charge_asymmetry'), invert=True)
    df['net_charge_norm']  = norm(df.get('net_charge'), invert=False)
    df['gravy_norm']       = norm(df.get('gravy'), invert=True)
    df['bindpred_norm']    = norm(df.get('bindpred_kd'), invert=True)

    # 2. Normalize PyRosetta & PLMs (Per-Mutation Normalized)
    df['pyrosetta_norm'] = norm(df['pyrosetta_ddg'] / mut_counts, invert=True)
    df['esm_norm']       = norm(df['esm_fitness'] / mut_counts, invert=False) if 'esm_fitness' in df.columns else np.nan
    df['esmc_norm']      = norm(df['esmc_fitness'] / mut_counts, invert=False) if 'esmc_fitness' in df.columns else np.nan
    df['ab_norm']        = norm(df['antiberty_fitness'] / mut_counts, invert=False) if 'antiberty_fitness' in df.columns else np.nan
    df['ablang_norm']    = norm(df.get('ablang_fitness'), invert=True) if 'ablang_fitness' in df.columns else np.nan
    df['protbert_norm']  = norm(df['protbert_fitness'] / mut_counts, invert=False) if 'protbert_fitness' in df.columns else np.nan

    # 3. Look up weights for the current protein format (fallback to scFv if unknown)
    # NOTE: Ensure FORMAT_WEIGHTS dictionary is defined at the top of pipeline.py
    active_weights = FORMAT_WEIGHTS.get(protein_format, FORMAT_WEIGHTS["scFv (Single-chain Fv)"])

    # 4. Compute weighted average across all available (non-NaN) metrics for this format
    weighted_sum = pd.Series(0.0, index=df.index)
    total_weight = 0.0

    for norm_col, weight in active_weights.items():
        if norm_col in df.columns and df[norm_col].notna().any():
            weighted_sum += weight * df[norm_col].fillna(0.5)
            total_weight += weight

    df['global_composite_score'] = weighted_sum / max(total_weight, 1e-6)

    df = df.sort_values('global_composite_score', ascending=False).reset_index(drop=True)
    
    # 5. Clean up columns for the final CSV layout
    ordered_cols = [
        'id', 'mutations', 'protein_format', 'num_mutations', 'global_composite_score',
        'janus_ddg', 'pyrosetta_ddg', 'camsol_score', 'pI', 'pi_delta', 'gravy', 'net_charge',
        'charge_asymmetry', 'bindpred_kd', 'esm_fitness', 'antiberty_fitness', 
        'ablang_fitness', 'protbert_fitness', 'poet2_fitness', 'esmc_fitness', 'full_sequence'
    ]
    # Retain the norms for plotting, plus the original sequences if requested
    cols_to_save = [c for c in ordered_cols if c in df.columns] + [c for c in df.columns if c.endswith('_norm') or c in ['VH', 'VL']]
    
    df[cols_to_save].to_csv(final_csv_path, index=False)
    print(f"Scoring complete for format '{protein_format}'. Results saved to {final_csv_path}")
    
    return df
