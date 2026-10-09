import os
import sys
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import subprocess
import shutil
import urllib.request
from transformers import AutoTokenizer, EsmForProteinFolding, AutoModelForMaskedLM
import pyrosetta
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
import antifold
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from catboost import CatBoostRegressor
from huggingface_hub import hf_hub_download
import gc
import glob
import streamlit as st
import esm

def score_scfv_variants(run_dir: str, run_name: str, wt_sequence: str):
    print(f"--- Phase 2: Scoring Variants for {run_name} ---")
    
    # ---------------------------------------------------------
    # 1. File Checks & Parsing
    # ---------------------------------------------------------
    raw_fasta_path = os.path.join(run_dir, f"{run_name}_relaxed_AB.fasta")
    camsol_txt_path = os.path.join(run_dir, f"{run_name}_camsol.txt")
    
    if not os.path.exists(raw_fasta_path):
        raise FileNotFoundError(f"Cannot find {raw_fasta_path}. Run Phase 1 first.")
    
    janus_src_path = "/home/esigalas/workspace/JanusDDG/src" 
    if janus_src_path not in sys.path:
        sys.path.append(janus_src_path)
        
    import utils

    utils.MODELS_DIR = "/home/esigalas/workspace/JanusDDG/models" 

    records = list(SeqIO.parse(raw_fasta_path, "fasta"))
    variants = []
    clean_wt = wt_sequence.replace(":", "").replace("/", "")
    
    for i, record in enumerate(records):
        unique_id = f"{run_name}_WT" if i == 0 else f"{run_name}_sample_{i}"
        raw_seq_str = str(record.seq).replace(":", "").replace("/", "")
        
        # Approximate VH/VL split based on WT length
        vh_len = len(clean_wt) // 2  # Adjust if your sequences split differently
        vh, vl = raw_seq_str[:vh_len], raw_seq_str[vh_len:]
            
        mut_seq = vh + vl
        mutations = []
        
        if mut_seq == clean_wt:
            mut_string = "WT"
        else:
            for idx, (wt_aa, mut_aa) in enumerate(zip(clean_wt, mut_seq)):
                if wt_aa != mut_aa:
                    mutations.append(f"{wt_aa}{idx+1}{mut_aa}")
            mut_string = ",".join(mutations)
            
        variants.append({"id": unique_id, "mutations": mut_string, "VH": vh, "VL": vl})
        
    df = pd.DataFrame(variants)
    
    # ---------------------------------------------------------
    # 2. Integrate Web Server CamSol Scores
    # ---------------------------------------------------------
    camsol_df = pd.read_csv(camsol_txt_path, sep='\t')
    df['camsol_score'] = camsol_df['protein variant score'].values
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wt_list = list(clean_wt)

    # ---------------------------------------------------------
    # 3. PLM 1: AntiBERTy (Antibody-Specific Fitness)
    # ---------------------------------------------------------
    print("Loading AntiBERTy...")
    ab_tokenizer = AutoTokenizer.from_pretrained("jeffruffolo/AntiBERTy")
    ab_model = AutoModelForMaskedLM.from_pretrained("jeffruffolo/AntiBERTy").to(device)
    ab_model.eval()

    ab_fitness_scores = []
    for _, row in df.iterrows():
        if row['mutations'] == "WT":
            ab_fitness_scores.append(0.0)
            continue
            
        mut_delta_logp = 0.0
        for m in row['mutations'].split(","):
            wt_aa, pos, mut_aa = m[0], int(m[1:-1]) - 1, m[-1]
            masked_seq = wt_list.copy()
            masked_seq[pos] = "[MASK]" 
            
            inputs = ab_tokenizer(" ".join(masked_seq), return_tensors="pt").to(device)
            with torch.no_grad():
                logits = ab_model(**inputs).logits
                log_probs = torch.log_softmax(logits[0, pos + 1], dim=-1) 
                mut_delta_logp += (log_probs[ab_tokenizer.convert_tokens_to_ids(mut_aa)] - 
                                   log_probs[ab_tokenizer.convert_tokens_to_ids(wt_aa)]).item()
        ab_fitness_scores.append(mut_delta_logp)
        
    df['antiberty_fitness'] = ab_fitness_scores
    del ab_model, ab_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 4. PLM 2: AbLang2 (Paired Germline Fitness)
    # ---------------------------------------------------------
    print("Loading AbLang2...")
    import ablang2
    import numpy as np
    
    # Initialized exactly as your snippet demonstrated
    ablang_model = ablang2.pretrained("ablang2-paired", random_init=False) 
    
    ablang_fitness_scores = []
    
    # 1. Calculate the baseline Wild-Type (WT) sequence likelihood
    wt_row = df[df['mutations'] == "WT"].iloc[0]
    
    # ablang returns an array of likelihoods for each position. We sum them for the total sequence score.
    wt_res = ablang_model([[wt_row['VH'], wt_row['VL']]], mode='likelihood')
    wt_score = np.sum(wt_res[0]) 
    
    for _, row in df.iterrows():
        if row['mutations'] == "WT":
            ablang_fitness_scores.append(0.0)
            continue
            
        # 2. Calculate the Mutant sequence likelihood
        mut_res = ablang_model([[row['VH'], row['VL']]], mode='likelihood')
        mut_score = np.sum(mut_res[0])
        
        # 3. Delta Log-Likelihood (Mutant - WT)
        # Positive values mean the mutant is more fit/germline-like than the WT
        mut_delta_logp = mut_score - wt_score
        
        ablang_fitness_scores.append(mut_delta_logp)

    df['ablang_fitness'] = ablang_fitness_scores
    
    # Safely clear the model from GPU
    del ablang_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 5. PLM 3: ProtBERT (Evolutionary Fitness)
    # ---------------------------------------------------------
    print("Loading ProtBERT...")
    
    pb_tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
    pb_model = AutoModelForMaskedLM.from_pretrained("Rostlab/prot_bert").to(device)
    pb_model.eval()

    pb_fitness_scores = []
    for _, row in df.iterrows():
        if row['mutations'] == "WT":
            pb_fitness_scores.append(0.0)
            continue
            
        mut_delta_logp = 0.0
        for m in row['mutations'].split(","):
            wt_aa, pos, mut_aa = m[0], int(m[1:-1]) - 1, m[-1]
            masked_seq = wt_list.copy()
            masked_seq[pos] = "[MASK]" 
            
            # ProtBERT requires space-separated sequences like AntiBERTy
            inputs = pb_tokenizer(" ".join(masked_seq), return_tensors="pt").to(device)
            
            with torch.no_grad():
                logits = pb_model(**inputs).logits
                # ProtBERT adds a [CLS] token at the start, so offset by 1
                log_probs = torch.log_softmax(logits[0, pos + 1], dim=-1)
                
                wt_tok = pb_tokenizer.convert_tokens_to_ids(wt_aa)
                mut_tok = pb_tokenizer.convert_tokens_to_ids(mut_aa)
                
                mut_delta_logp += (log_probs[mut_tok] - log_probs[wt_tok]).item()
                
        pb_fitness_scores.append(mut_delta_logp)

    df['protbert_fitness'] = pb_fitness_scores
    del pb_model, pb_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 6. PLM 4: ESM-2 (General Fitness) & BINDPRED
    # ---------------------------------------------------------
    print("Loading ESM-2 & BINDPRED...")
    model_esm, alphabet_esm = esm.pretrained.esm2_t33_650M_UR50D()
    model_esm = model_esm.to(device)
    batch_converter_esm = alphabet_esm.get_batch_converter()
    model_esm.eval()

    bindpred_model = CatBoostRegressor()
    bindpred_model.load_model(hf_hub_download(repo_id="hbp5181/BindPred", filename="ESM2_BindPred.cbm"), format="cbm")

    net_charges, bindpred_scores, esm_fitness_scores = [], [], []

    for _, row in df.iterrows():
        net_charges.append(ProteinAnalysis(row['VH'] + row['VL']).charge_at_pH(7.4))
        
        _, _, batch_tokens = batch_converter_esm([("VH", row['VH']), ("VL", row['VL'])])
        with torch.no_grad():
            reps = model_esm(batch_tokens.to(device), repr_layers=[33], return_contacts=False)["representations"][33]
            bindpred_scores.append(bindpred_model.predict(np.hstack([
                reps[0, 1:len(row['VH'])+1].mean(0).cpu().numpy(),
                reps[1, 1:len(row['VL'])+1].mean(0).cpu().numpy()
            ])))

        if row['mutations'] == "WT":
            esm_fitness_scores.append(0.0)
        else:
            mut_delta_logp = 0.0
            for m in row['mutations'].split(","):
                wt_aa, pos, mut_aa = m[0], int(m[1:-1]) - 1, m[-1]
                masked_seq = wt_list.copy()
                masked_seq[pos] = "<mask>"
                
                _, _, mask_tokens = batch_converter_esm([("seq", "".join(masked_seq))])
                with torch.no_grad():
                    log_probs = torch.log_softmax(model_esm(mask_tokens.to(device))["logits"][0, pos + 1], dim=-1)
                    mut_delta_logp += (log_probs[alphabet_esm.tok_to_idx[mut_aa]] - log_probs[alphabet_esm.tok_to_idx[wt_aa]]).item()
            esm_fitness_scores.append(mut_delta_logp)

    df['net_charge'], df['bindpred_kd'], df['esm_fitness'] = net_charges, bindpred_scores, esm_fitness_scores

    # ---------------------------------------------------------
    # 7. JanusDDG Scoring
    # ---------------------------------------------------------
    print("Running JanusDDG...")
    temp_csv_path = os.path.join(run_dir, "temp_janus_input.csv")
    pd.DataFrame({'ID': df[df['mutations'] != "WT"]['id'], 'Sequence': clean_wt, 
                  'MTS': df[df['mutations'] != "WT"]['mutations'].str.replace(",", "_")}).to_csv(temp_csv_path, index=False)
    
    import __main__
    import model as janus_model
    janus_model.device = device
    setattr(__main__, 'Cross_Attention_DDG', janus_model.Cross_Attention_DDG)
    setattr(__main__, 'TransformerRegression', janus_model.TransformerRegression)
    setattr(__main__, 'SinusoidalPositionalEncoding', janus_model.SinusoidalPositionalEncoding)

    model_janus = utils.load_model('JanusDDG_fine_tuned.pth', device)
    pred_dir, _ = utils.process_and_predict(temp_csv_path, model_janus, model_esm, batch_converter_esm, device)
    df = df.merge(pd.DataFrame({'id': df[df['mutations'] != "WT"]['id'], 'janus_ddg': pred_dir.values}), on='id', how='left').fillna({'janus_ddg': 0.0})
    os.remove(temp_csv_path)

    # ---------------------------------------------------------
    # 8. 7-Pillar Normalization & Global Composite Score
    # ---------------------------------------------------------
    df['survives_ddg'] = df['janus_ddg'] >= -1.0
    df['survives_camsol'] = df['camsol_score'] >= df['camsol_score'].quantile(0.10)
    df['survival_status'] = df['survives_ddg'] & df['survives_camsol']
    
    def norm(col, invert=False):
        c_min, c_max = df[col].min(), df[col].max()
        if c_max == c_min: return 0.5
        return (c_max - df[col]) / (c_max - c_min) if invert else (df[col] - c_min) / (c_max - c_min)

    df['camsol_norm'] = norm('camsol_score')
    df['janus_norm'] = norm('janus_ddg', invert=True)     # Lower is better
    df['bindpred_norm'] = norm('bindpred_kd', invert=True) # Lower is better
    df['esm_norm'] = norm('esm_fitness')
    df['ab_norm'] = norm('antiberty_fitness')
    df['ablang_norm'] = norm('ablang_fitness')
    df['protbert_norm'] = norm('protbert_fitness')

    # Update the global composite score equation:
    df['global_composite_score'] = (df['camsol_norm'] + df['janus_norm'] + df['bindpred_norm'] + 
                                    df['esm_norm'] + df['ab_norm'] + df['ablang_norm'] + df['protbert_norm']) / 7.0
    
    df = df.sort_values('global_composite_score', ascending=False).reset_index(drop=True)
    df.to_csv(os.path.join(run_dir, f"{run_name}_developability_metrics.csv"), index=False)
    
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return df

def run_scfv_developability_pipeline(vh_seq: str, vl_seq: str, base_output_dir: str, run_name: str, 
                                     gen_models=["AntiFold (Antibody-specific)"], num_seqs=100, temp=0.2):
    
    # 1. Create a dedicated workspace for this specific run
    run_dir = os.path.join(base_output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # 2. Dynamically prefix all output files with the run_name
    unrelaxed_pdb_path = os.path.join(run_dir, f"{run_name}_unrelaxed.pdb")
    relaxed_pdb_path = os.path.join(run_dir, f"{run_name}_relaxed.pdb")

    print(f"Starting pipeline for: {run_name}")
    print(f"Workspace initialized at: {run_dir}")
    
    # ---------------------------------------------------------
    # STEP 1: Input Formatting (Fv Dimer Setup)
    # ---------------------------------------------------------
    dimer_seq = f"{vh_seq}:{vl_seq}"    
    print(f"Step 1 Complete: Formatted dimer sequence -> {dimer_seq}")

    # ---------------------------------------------------------
    # STEP 2: 3D Structure Prediction (Local ESMFold with Multimer Fix)
    # ---------------------------------------------------------
    if os.path.exists(unrelaxed_pdb_path):
        print(f"Step 2: Found existing {unrelaxed_pdb_path}. Skipping ESMFold prediction...")
    else:
        print("Step 2: Running Structure Prediction...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        
        model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1",
            torch_dtype=torch.bfloat16,
            device_map="auto" 
        )
        model.trunk.set_chunk_size(64)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        combined_seq = vh_seq + vl_seq
        
        tokenized_input = tokenizer(
            combined_seq, 
            return_tensors="pt", 
            add_special_tokens=False, 
            return_token_type_ids=False
        ).to(device)
        
        seq_length = len(combined_seq)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids[len(vh_seq):] += 512 
        tokenized_input["position_ids"] = position_ids.unsqueeze(0)
        
        with torch.no_grad():
            output = model(**tokenized_input)
            
            float32_output = {
                k: v.to(torch.float32) if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16 else v 
                for k, v in output.items()
            }
            
            pdb_string = model.output_to_pdb(float32_output)[0]
            
        pdb_lines = pdb_string.split("\n")
        fixed_pdb = []
        vh_len = len(vh_seq)
        previous_chain = 'A'
        
        for line in pdb_lines:
            if line.startswith("ATOM"):
                res_idx = int(line[22:26].strip())
                current_chain = 'A'
                
                if res_idx > vh_len:
                    current_chain = 'B'
                    line = line[:21] + 'B' + line[22:]
                    # FIX: Subtract both the VH length AND the artificial 512 offset
                    new_res_idx = res_idx - vh_len - 512
                    line = line[:22] + f"{new_res_idx:4d}" + line[26:]
                
                if current_chain != previous_chain:
                    fixed_pdb.append("TER")
                    previous_chain = current_chain
                    
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
        pyrosetta.init("-relax:constrain_relax_to_start_coords -ex1 -ex2 -use_input_sc")
        
        pose = pyrosetta.pose_from_file(unrelaxed_pdb_path)
        scorefxn = ScoreFunctionFactory.create_score_function("ref2015")
        
        relax = FastRelax()
        relax.set_scorefxn(scorefxn)
        relax.constrain_relax_to_start_coords(True) 
        relax.apply(pose)
        
        pose.dump_pdb(relaxed_pdb_path)

    # ---------------------------------------------------------
    # STEP 4: Multi-Model Generative Ensemble
    # ---------------------------------------------------------    

    print(f"Step 4: Running Generative Ensemble: {gen_models}")
    
    all_raw_records = []
    
    # --- MODEL A: AntiFold ---
    if "AntiFold (Antibody-specific)" in gen_models:
        print(" -> Running AntiFold at Temperature {temp}...")
        
        antifold_cmd = [
            "python", "-m", "antifold.main",
            "--pdb_file", relaxed_pdb_path,
            "--heavy_chain", "A", "--light_chain", "B",
            "--regions", "FWL FWH",
            "--num_seq_per_target", str(num_seqs),
            "--sampling_temp", str(temp),
            "--out_dir", run_dir
        ]
        try:
            # capture_output forces the internal errors to be saved so we can read them
            subprocess.run(antifold_cmd, check=True, capture_output=True, text=True)
            af_fasta = os.path.join(run_dir, f"{run_name}_relaxed_AB.fasta")
            if os.path.exists(af_fasta):
                af_records = list(SeqIO.parse(af_fasta, "fasta"))
                SeqIO.write(af_records, os.path.join(run_dir, f"{run_name}_antifold.fasta"), "fasta")
                all_raw_records.extend(af_records)

        except subprocess.CalledProcessError as e:
            # Route the actual crash log to the Streamlit UI and stop the pipeline
            st.error(f"**AntiFold Crashed!**\n\nError Log:\n```\n{e.stderr}\n```")
            raise RuntimeError("Phase 1 halted due to AntiFold failure.")

    # --- DYNAMIC CDR MASKING FOR MPNN MODELS ---
    # FIX 1: Change file extension to .jsonl
    fixed_json_path = os.path.join(run_dir, "fixed_positions.jsonl")
    
    if ("ProteinMPNN (Diverse)" in gen_models) or ("ThermoMPNN (ddG-Driven)" in gen_models) or ("ProteinMPNN (Soluble)" in gen_models) or ("ESM-C (Evolutionary LM)" in gen_models) or ("AbLang2 (Antibody LM)" in gen_models):
        from abnumber import Chain
        import json
        
        vh_chain = Chain(vh_seq, scheme='chothia')
        vl_chain = Chain(vl_seq, scheme='chothia')
        
        fixed_A, fixed_B = [], []
        
        for cdr in [vh_chain.cdr1_seq, vh_chain.cdr2_seq, vh_chain.cdr3_seq]:
            start = vh_seq.find(cdr)
            if start != -1:
                fixed_A.extend(list(range(start + 1, start + 1 + len(cdr))))
                
        for cdr in [vl_chain.cdr1_seq, vl_chain.cdr2_seq, vl_chain.cdr3_seq]:
            start = vl_seq.find(cdr)
            if start != -1:
                fixed_B.extend(list(range(start + 1, start + 1 + len(cdr))))
                
        pdb_basename = os.path.splitext(os.path.basename(relaxed_pdb_path))[0]
        with open(fixed_json_path, 'w') as f:
            json.dump({pdb_basename: {"A": fixed_A, "B": fixed_B}}, f)

    # --- MODEL B: ProteinMPNN ---
    if "ProteinMPNN (Diverse)" in gen_models:
        print(f" -> Running ProteinMPNN at Temperature {temp}...")
        mpnn_out_dir = os.path.join(run_dir, "mpnn_out")
        
        PROTEIN_MPNN_SCRIPT = "/home/esigalas/workspace/ProteinMPNN/protein_mpnn_run.py"
        
        if not os.path.exists(PROTEIN_MPNN_SCRIPT):
            st.error(f"Could not find ProteinMPNN at {PROTEIN_MPNN_SCRIPT}.")
            raise FileNotFoundError("Missing ProteinMPNN script.")
            
        mpnn_cmd = [
            "python", PROTEIN_MPNN_SCRIPT,
            "--pdb_path", relaxed_pdb_path,
            "--pdb_path_chains", "A B",
            # FIX 2: Removed --chains_to_design entirely
            # FIX 3: Corrected flag to --fixed_positions_jsonl
            "--fixed_positions_jsonl", fixed_json_path,
            "--out_folder", mpnn_out_dir,
            "--num_seq_per_target", str(num_seqs),
            "--sampling_temp", str(temp),
            "--batch_size", "1"
        ]
        try:
            subprocess.run(mpnn_cmd, check=True, capture_output=True, text=True)
            mpnn_fastas = glob.glob(os.path.join(mpnn_out_dir, "seqs", "*.fa"))
            
            mpnn_records = []
            for fa in mpnn_fastas:
                mpnn_records.extend(list(SeqIO.parse(fa, "fasta")))
                
            if mpnn_records:
                SeqIO.write(mpnn_records, os.path.join(run_dir, f"{run_name}_proteinmpnn.fasta"), "fasta")
                all_raw_records.extend(mpnn_records)

        except subprocess.CalledProcessError as e:
            st.error(f"**ProteinMPNN Crashed!**\n\nError Log:\n```\n{e.stderr}\n```")
            raise RuntimeError("Phase 1 halted due to ProteinMPNN failure.")

    # --- MODEL C: ThermoMPNN (ddG-Driven) ---
    if "ThermoMPNN (ddG-Driven)" in gen_models:
        print(" -> Running ThermoMPNN ddG Generator (Full Complex)...")
        
        THERMO_MPNN_SCRIPT = "/home/esigalas/workspace/ThermoMPNN/analysis/custom_inference.py"
        
        if not os.path.exists(THERMO_MPNN_SCRIPT):
            st.error(f"Could not find ThermoMPNN at {THERMO_MPNN_SCRIPT}.")
            st.stop()
            
        pdb_basename = os.path.splitext(os.path.basename(relaxed_pdb_path))[0]
        csv_path = os.path.join(run_dir, f"ThermoMPNN_inference_{pdb_basename}.csv")
        
        all_thermo_muts = []
        clean_wt = vh_seq + vl_seq
        
        # Combine the 1-indexed PyRosetta coordinates into a single 0-indexed flat set for CSV matching
        combined_cdr_mask = {p - 1 for p in fixed_A}.union({len(vh_seq) + p - 1 for p in fixed_B})
        
        # Pass "AB" so alt_parse_PDB parses the whole complex into a single graph
        thermo_cmd = [
            "python", THERMO_MPNN_SCRIPT,
            "--pdb", relaxed_pdb_path,
            "--chain", "AB", 
            "--out_dir", run_dir
        ]
        
        try:
            subprocess.run(thermo_cmd, check=True, capture_output=True, text=True)
            
            if not os.path.exists(csv_path):
                st.error("ThermoMPNN failed to create output CSV.")
                st.stop()
                
            df_thermo = pd.read_csv(csv_path)
            
            # Parse the unified CSV
            for _, row in df_thermo.iterrows():
                pos = int(row['position']) 
                mut_aa = row['mutation']
                ddg = float(row['ddG_pred'])
                
                if pos in combined_cdr_mask:
                    continue 
                    
                # We want stabilizing mutations (negative ddG)
                if ddg < 0:
                    all_thermo_muts.append({
                        'pos': pos,
                        'wt_aa': row['wildtype'],
                        'mut_aa': mut_aa,
                        'ddg': ddg
                    })
                    
            os.remove(csv_path)
            
        except subprocess.CalledProcessError as e:
            st.error(f"**ThermoMPNN Crashed!**\n\nError Log:\n```\n{e.stderr}\n```")
            st.stop()
            
        # Sort by most stabilizing (most negative ddG)
        all_thermo_muts = sorted(all_thermo_muts, key=lambda x: x['ddg'])
        top_muts = all_thermo_muts[:int(num_seqs)]
        
        print(f"    Building {len(top_muts)} guaranteed stabilizing ThermoMPNN candidates...")
        
        thermo_records = []
        for i, m in enumerate(top_muts):
            mut_seq = list(clean_wt)
            mut_seq[m['pos']] = m['mut_aa']
            
            new_record = SeqRecord(
                Seq("".join(mut_seq)), 
                id=f"ThermoMPNN_{m['wt_aa']}{m['pos']+1}{m['mut_aa']}_rank{i+1}", 
                description=""
            )
            thermo_records.append(new_record)
            if thermo_records:
                SeqIO.write(thermo_records, os.path.join(run_dir, f"{run_name}_thermompnn.fasta"), "fasta")
                all_raw_records.extend(thermo_records)
    
    # --- MODEL D: ProteinMPNN (Soluble) ---
    if "ProteinMPNN (Soluble)" in gen_models:
        print(f" -> Running ProteinMPNN (Soluble Model) at Temperature {temp}...")
        soluble_out_dir = os.path.join(run_dir, "soluble_mpnn_out")
        
        PROTEIN_MPNN_SCRIPT = "/home/esigalas/workspace/ProteinMPNN/protein_mpnn_run.py"
        
        soluble_cmd = [
            "python", PROTEIN_MPNN_SCRIPT,
            "--pdb_path", relaxed_pdb_path,
            "--pdb_path_chains", "A B",
            "--fixed_positions_jsonl", fixed_json_path,
            "--out_folder", soluble_out_dir,
            "--num_seq_per_target", str(num_seqs),
            "--sampling_temp", str(temp),
            "--batch_size", "1",
            "--use_soluble_model"  # The flag that triggers the specialized weights
        ]
        try:
            subprocess.run(soluble_cmd, check=True, capture_output=True, text=True)
            soluble_fastas = glob.glob(os.path.join(soluble_out_dir, "seqs", "*.fa"))
            
            soluble_records = []
            for fa in soluble_fastas:
                soluble_records.extend(list(SeqIO.parse(fa, "fasta")))
                
            # Save a standalone SolubleMPNN FASTA
            if soluble_records:
                SeqIO.write(soluble_records, os.path.join(run_dir, f"{run_name}_solublempnn.fasta"), "fasta")
                all_raw_records.extend(soluble_records)
                
        except subprocess.CalledProcessError as e:
            st.error(f"**ProteinMPNN (Soluble) Crashed!**\n\nError Log:\n```\n{e.stderr}\n```")
            st.stop()
    
    # --- MODEL E: ESM-C (Isolated Environment Subprocess) ---
    if "ESM-C (Evolutionary LM)" in gen_models:
        print(" -> Running ESM-C Masked-Marginal Probability Generator (Isolated Env)...")
        esmc_fasta = os.path.join(run_dir, f"{run_name}_esmc.fasta")
        
        # Combine CDR mask into a comma-separated string for the subprocess argument
        combined_cdr_mask = {p - 1 for p in fixed_A}.union({len(vh_seq) + p - 1 for p in fixed_B})
        cdr_str = ",".join(map(str, combined_cdr_mask))
        clean_wt = vh_seq + vl_seq
        
        # Use conda run to execute the script entirely within the Python 3.10 micro-environment
        esmc_cmd = [
            "conda", "run", "-n", "esmc_env", 
            "python", "run_esmc_dms.py",
            "--sequence", clean_wt,
            "--cdr_indices", cdr_str,
            "--num_seqs", str(num_seqs),
            "--out_fasta", esmc_fasta
        ]
        
        try:
            subprocess.run(esmc_cmd, check=True, capture_output=True, text=True)
            
            if os.path.exists(esmc_fasta):
                esmc_records = list(SeqIO.parse(esmc_fasta, "fasta"))
                all_raw_records.extend(esmc_records)
                
        except subprocess.CalledProcessError as e:
            st.error(f"**ESM-C Subprocess Crashed!**\n\nError Log:\n```\n{e.stderr}\n```")
            st.stop()
    
    # --- MODEL F: AbLang2-Paired (Native OPIG Installation) ---
    if "AbLang2 (Antibody LM)" in gen_models:
        print(" -> Running AbLang2-Paired Masked-Marginal Probability Generator...")
        import ablang2
        import numpy as np
        
        try:
            # Load your working, local OPIG paired model
            ablang = ablang2.pretrained(model_to_use="ablang2-paired", random_init=False)
            
            # AbLang2 alphabet order (standard 20 AAs)
            alphabet = list("ACDEFGHIKLMNPQRSTVWY")
            
            vh_len = len(vh_seq)
            combined_cdr_mask = {p - 1 for p in fixed_A}.union({vh_len + p - 1 for p in fixed_B})
            
            ablang_muts = []
            
            # Iterate over the Heavy Chain (VH)
            for i in range(len(vh_seq)):
                if i in combined_cdr_mask: continue
                
                wt_aa = vh_seq[i]
                # AbLang2 expects the mask to be an asterisk
                masked_vh = vh_seq[:i] + "*" + vh_seq[i+1:]
                
                # Forward pass requesting normalized probabilities
                res = ablang([[masked_vh, vl_seq]], mode='probability')
                
                # FIX: Removed the extra [0]. 
                # res[0] is the 2D array [seq_len, 20]. res[0][i] gets the 20 probabilities for position i.
                probs = res[0][i] 
                
                wt_idx = alphabet.index(wt_aa) if wt_aa in alphabet else None
                if wt_idx is None: continue
                wt_prob = probs[wt_idx]
                
                for mut_idx, mut_aa in enumerate(alphabet):
                    if mut_aa == wt_aa: continue
                    mut_prob = probs[mut_idx]
                    
                    # Calculate probability difference
                    delta_p = mut_prob - wt_prob
                    if delta_p > 0:
                        ablang_muts.append({
                            'pos': i, 'wt_aa': wt_aa, 'mut_aa': mut_aa, 'delta': delta_p
                        })

            # Iterate over the Light Chain (VL)
            for i in range(len(vl_seq)):
                global_pos = vh_len + i
                if global_pos in combined_cdr_mask: continue
                
                wt_aa = vl_seq[i]
                masked_vl = vl_seq[:i] + "*" + vl_seq[i+1:]
                
                res = ablang([[vh_seq, masked_vl]], mode='probability')
                
                # FIX: Removed the extra [0] here as well
                probs = res[0][global_pos]
                
                wt_idx = alphabet.index(wt_aa) if wt_aa in alphabet else None
                if wt_idx is None: continue
                wt_prob = probs[wt_idx]
                
                for mut_idx, mut_aa in enumerate(alphabet):
                    if mut_aa == wt_aa: continue
                    mut_prob = probs[mut_idx]
                    
                    delta_p = mut_prob - wt_prob
                    if delta_p > 0:
                        ablang_muts.append({
                            'pos': global_pos, 'wt_aa': wt_aa, 'mut_aa': mut_aa, 'delta': delta_p
                        })

            # Sort by highest evolutionary probability boost
            ablang_muts = sorted(ablang_muts, key=lambda x: x['delta'], reverse=True)
            top_muts = ablang_muts[:int(num_seqs)]
            
            print(f"    Building {len(top_muts)} masked evolutionary candidates from AbLang2-Paired...")
            
            clean_wt = vh_seq + vl_seq
            ablang_records = []
            for i, m in enumerate(top_muts):
                mut_seq = list(clean_wt)
                mut_seq[m['pos']] = m['mut_aa']
                new_record = SeqRecord(
                    Seq("".join(mut_seq)), 
                    id=f"AbLang2Paired_{m['wt_aa']}{m['pos']+1}{m['mut_aa']}_rank{i+1}", 
                    description=""
                )
                ablang_records.append(new_record)
            
            if ablang_records:
                SeqIO.write(ablang_records, os.path.join(run_dir, f"{run_name}_ablang2.fasta"), "fasta")
                all_raw_records.extend(ablang_records)
                
        except Exception as e:
            st.error(f"**AbLang2-Paired Crashed!**\n\nError Log:\n```\n{str(e)}\n```")
            st.stop()

    # ---------------------------------------------------------
    # STEP 5: Pooling, Deduplication, and CamSol Formatting
    # ---------------------------------------------------------
    camsol_fasta_path = os.path.join(run_dir, f"{run_name}_camsol_upload.fasta")
    print(f"Step 5: Pooling candidates and formatting for CamSol -> {camsol_fasta_path}")
    
    unique_seqs = {}
    cleaned_records = []
    
    # 1. Isolate the Wild-Type sequence first to ensure it's always at index 0
    clean_wt = (vh_seq + vl_seq).replace(":", "").replace("/", "")
    
    wt_record = SeqRecord(Seq(clean_wt), id=f"{run_name}_WT", description="")
    cleaned_records.append(wt_record)
    unique_seqs[clean_wt] = True 
    
    # 2. Iterate through all generated records across all models
    for record in all_raw_records:
        clean_seq = str(record.seq).replace(":", "").replace("/", "")
        
        # Deduplicate: Only add if we haven't seen this exact sequence before
        if clean_seq not in unique_seqs:
            unique_seqs[clean_seq] = True
            
            # Assign unique sequential ID (e.g., sample_1, sample_2)
            new_record = SeqRecord(
                Seq(clean_seq), 
                id=f"{run_name}_sample_{len(cleaned_records)}", 
                description=""
            )
            cleaned_records.append(new_record)

    # 3. Save the final merged payload
    SeqIO.write(cleaned_records, camsol_fasta_path, "fasta")
    print(f"Step 5 Complete: Pooled {len(cleaned_records)} unique variants across {len(gen_models)} models.")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
