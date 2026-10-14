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
import json

def score_scfv_variants(run_dir: str, run_name: str, wt_sequence: str):
    print(f"--- Phase 2: Scoring Variants for {run_name} ---")
    
    # ---------------------------------------------------------
    # 1. Initialization & FASTA Parsing
    # ---------------------------------------------------------
    raw_fasta_path = os.path.join(run_dir, f"{run_name}_camsol_upload.fasta")
    camsol_txt_path = os.path.join(run_dir, f"{run_name}_camsol.txt")
    final_csv_path = os.path.join(run_dir, f"{run_name}_developability_metrics.csv")
    metadata_path = os.path.join(run_dir, f"{run_name}_metadata.json")
    
    if not os.path.exists(raw_fasta_path):
        raise FileNotFoundError(f"Cannot find {raw_fasta_path}. Run Phase 1 first.")
    
    # --- LOAD METADATA ---
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata missing. Please re-run Phase 1 for {run_name} to generate the manifest.")
        
    with open(metadata_path, "r") as f:
        meta = json.load(f)
        vh_len = meta["vh_len"]
        
    print(f" -> Loaded run manifest. Splitting domains at VH length: {vh_len} aa")

    records = list(SeqIO.parse(raw_fasta_path, "fasta"))
    clean_wt = wt_sequence.replace(":", "").replace("/", "")
    
    variants = []
    master_fasta_mapping = []
    
    for i, record in enumerate(records):
        unique_id = f"{run_name}_WT" if i == 0 else f"{run_name}_sample_{i}"
        seq = str(record.seq).replace(":", "").replace("/", "")
        
        master_fasta_mapping.append({"id": unique_id, "seq": seq})
        
        vh, vl = seq[:vh_len], seq[vh_len:]
        mutations = []
        if seq == clean_wt:
            mut_string = "WT"
        else:
            for idx, (wt_aa, mut_aa) in enumerate(zip(clean_wt, seq)):
                if wt_aa != mut_aa:
                    mutations.append(f"{wt_aa}{idx+1}{mut_aa}")
            mut_string = ",".join(mutations)
            
        variants.append({"id": unique_id, "mutations": mut_string, "VH": vh, "VL": vl})
        
    df = pd.DataFrame(variants)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wt_list = list(clean_wt)

    # ---------------------------------------------------------
    # 2. METRIC-LEVEL SMART CACHE MERGE
    # ---------------------------------------------------------
    if os.path.exists(final_csv_path):
        try:
            cached_df = pd.read_csv(final_csv_path)
            cols_to_drop = [c for c in cached_df.columns if 'norm' in c or 'survives' in c or 'global' in c or 'pi_delta' in c]
            cached_df = cached_df.drop(columns=cols_to_drop, errors='ignore')
            
            df = df.merge(cached_df.drop(columns=['mutations', 'VH', 'VL'], errors='ignore'), on='id', how='left')
            print(f" -> Cache loaded. Evaluating missing metrics for {len(df)} sequences...")
            
            # --- FAST CAMSOL UPDATE ---
            if 'camsol_score' in df.columns and df['camsol_score'].isna().all() and os.path.exists(camsol_txt_path):
                # Check if all other metrics are present (meaning ONLY camsol is missing)
                expected_compute_metrics = ['janus_ddg', 'pyrosetta_ddg', 'bindpred_kd', 'net_charge', 'pI', 'gravy', 'charge_asymmetry', 'esm_fitness', 'antiberty_fitness', 'ablang_fitness', 'protbert_fitness']
                if all(m in df.columns for m in expected_compute_metrics) and not df[expected_compute_metrics].isna().any().any():
                    print(" -> Cache Hit for all sequences! New CamSol data detected. Performing fast update...")
                    camsol_df = pd.read_csv(camsol_txt_path, sep='\t')
                    if len(camsol_df) == len(master_fasta_mapping):
                        camsol_mapping = {master_fasta_mapping[i]["id"]: score for i, score in enumerate(camsol_df['protein variant score'].values)}
                        df['camsol_score'] = df['id'].map(camsol_mapping)
                        
                        def norm_cache(col, invert=False):
                            if col not in df.columns or df[col].isna().all(): return 0.5 
                            c_min, c_max = df[col].min(), df[col].max()
                            if c_max == c_min: return 0.5
                            return (c_max - df[col]) / (c_max - c_min) if invert else (df[col] - c_min) / (c_max - c_min)
                            
                        df['camsol_norm'] = norm_cache('camsol_score')
                        df['janus_norm'] = norm_cache('janus_ddg', invert=True)      
                        df['pyrosetta_norm'] = norm_cache('pyrosetta_ddg', invert=True) 
                        df['bindpred_norm'] = norm_cache('bindpred_kd', invert=True) 
                        df['esm_norm'] = norm_cache('esm_fitness')
                        df['ab_norm'] = norm_cache('antiberty_fitness')
                        df['ablang_norm'] = norm_cache('ablang_fitness')
                        df['protbert_norm'] = norm_cache('protbert_fitness')
                        df['gravy_norm'] = norm_cache('gravy', invert=True)
                        df['charge_asym_norm'] = norm_cache('charge_asymmetry', invert=True)
                        
                        df['pi_delta'] = (df['pI'] - 7.4).abs()
                        df['pi_norm'] = norm_cache('pi_delta', invert=False) 
                        
                        score_sum = (df['janus_norm'] + df['pyrosetta_norm'] + df['bindpred_norm'] + 
                                     df['esm_norm'] + df['ab_norm'] + df['ablang_norm'] + 
                                     df['protbert_norm'] + df['gravy_norm'] + df['charge_asym_norm'] + 
                                     df['pi_norm'] + df['camsol_norm'])
                        
                        df['global_composite_score'] = score_sum / 11.0
                        
                        df = df.sort_values('global_composite_score', ascending=False).reset_index(drop=True)
                        cols_to_save = [c for c in df.columns if not c.endswith('_norm') and c != 'pi_delta']
                        df[cols_to_save].to_csv(final_csv_path, index=False)
                        print(" -> Fast update complete! Returning updated cache.")
                        return df
        except Exception as e:
            print(f" -> Could not verify cache: {e}. Recomputing from scratch...")
            
    # ---------------------------------------------------------
    # 3. Web Server CamSol Scores
    # ---------------------------------------------------------
    if 'camsol_score' not in df.columns: df['camsol_score'] = np.nan
        
    if df['camsol_score'].isna().any() and os.path.exists(camsol_txt_path):
        try:
            camsol_df = pd.read_csv(camsol_txt_path, sep='\t')
            if len(camsol_df) == len(master_fasta_mapping):
                camsol_mapping = {master_fasta_mapping[i]["id"]: score for i, score in enumerate(camsol_df['protein variant score'].values)}
                df['camsol_score'] = df['id'].map(camsol_mapping)
                print(" -> Integrated missing CamSol data.")
        except Exception as e:
            print(f" -> WARNING: Failed to parse CamSol: {e}")

    # ---------------------------------------------------------
    # 4. Fast Sequence Biophysics (pI, GRAVY, Charge Asymmetry, Net Charge)
    # ---------------------------------------------------------
    if 'pI' not in df.columns: df['pI'] = np.nan
    if 'gravy' not in df.columns: df['gravy'] = np.nan
    if 'charge_asymmetry' not in df.columns: df['charge_asymmetry'] = np.nan
    if 'net_charge' not in df.columns: df['net_charge'] = np.nan
    
    mask = df['pI'].isna() | df['gravy'].isna() | df['charge_asymmetry'].isna() | df['net_charge'].isna()
    
    if mask.any():
        print(f"Calculating basic sequence biophysics for {mask.sum()} missing variants...")
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        
        for idx, row in df[mask].iterrows():
            vh_seq, vl_seq = row['VH'], row['VL']
            full_seq = vh_seq + vl_seq
            
            vh_analysis = ProteinAnalysis(vh_seq)
            vl_analysis = ProteinAnalysis(vl_seq)
            full_analysis = ProteinAnalysis(full_seq)
            
            df.at[idx, 'net_charge'] = full_analysis.charge_at_pH(7.4)
            df.at[idx, 'pI'] = full_analysis.isoelectric_point()
            df.at[idx, 'gravy'] = full_analysis.gravy()
            
            # Asymmetry is the absolute difference in charge between the two domains
            df.at[idx, 'charge_asymmetry'] = abs(vh_analysis.charge_at_pH(7.4) - vl_analysis.charge_at_pH(7.4))
    else:
        print(" -> Sequence biophysics up to date. Skipping calculation.")

    # ---------------------------------------------------------
    # 5. PyRosetta Structural Thermodynamic Stability (ddG)
    # ---------------------------------------------------------
    if 'pyrosetta_ddg' not in df.columns: df['pyrosetta_ddg'] = np.nan
    mask = df['pyrosetta_ddg'].isna()
    
    if mask.any():
        print(f"Calculating PyRosetta for {mask.sum()} missing variants...")
        import pyrosetta
        from pyrosetta.toolbox import mutate_residue
        pyrosetta.init("-mute all")
        
        relaxed_pdb_path = os.path.join(run_dir, f"{run_name}_relaxed.pdb")
        if os.path.exists(relaxed_pdb_path):
            wt_pose = pyrosetta.pose_from_file(relaxed_pdb_path)
            scorefxn = pyrosetta.get_fa_scorefxn()
            wt_energy = scorefxn(wt_pose)
            
            for idx, row in df[mask].iterrows():
                if row['mutations'] == "WT":
                    df.at[idx, 'pyrosetta_ddg'] = 0.0
                    continue
                    
                mut_pose = wt_pose.clone()
                for m in row['mutations'].split(","):
                    pos, mut_aa = int(m[1:-1]), m[-1]
                    mutate_residue(mut_pose, pos, mut_aa, pack_scorefxn=scorefxn)
                    
                df.at[idx, 'pyrosetta_ddg'] = scorefxn(mut_pose) - wt_energy
        else:
            df['pyrosetta_ddg'] = 0.0
    else:
        print(" -> PyRosetta metrics up to date. Skipping model.")

    # ---------------------------------------------------------
    # 6. PLM 1: AntiBERTy (Antibody-Specific Fitness)
    # ---------------------------------------------------------
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
                wt_aa, pos, mut_aa = m[0], int(m[1:-1]) - 1, m[-1]
                masked_seq = wt_list.copy()
                masked_seq[pos] = "[MASK]" 
                
                inputs = ab_tokenizer(" ".join(masked_seq), return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = ab_model(**inputs).logits
                    log_probs = torch.log_softmax(logits[0, pos + 1], dim=-1) 
                    mut_delta_logp += (log_probs[ab_tokenizer.convert_tokens_to_ids(mut_aa)] - 
                                       log_probs[ab_tokenizer.convert_tokens_to_ids(wt_aa)]).item()
            df.at[idx, 'antiberty_fitness'] = mut_delta_logp
            
        del ab_model, ab_tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(" -> AntiBERTy metrics up to date. Skipping model.")

    # ---------------------------------------------------------
    # 7. PLM 2: AbLang2 (Paired Germline Fitness)
    # ---------------------------------------------------------
    if 'ablang_fitness' not in df.columns: df['ablang_fitness'] = np.nan
    mask = df['ablang_fitness'].isna()
    
    if mask.any():
        print(f"Loading AbLang2 to score {mask.sum()} missing variants...")
        import ablang2
        ablang_model = ablang2.pretrained("ablang2-paired", random_init=False) 
        
        wt_res = ablang_model([[clean_wt[:vh_len], clean_wt[vh_len:]]], mode='likelihood')
        wt_score = np.sum(wt_res[0]) 
        
        for idx, row in df[mask].iterrows():
            if row['mutations'] == "WT":
                df.at[idx, 'ablang_fitness'] = 0.0
                continue
                
            mut_res = ablang_model([[row['VH'], row['VL']]], mode='likelihood')
            df.at[idx, 'ablang_fitness'] = np.sum(mut_res[0]) - wt_score

        del ablang_model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(" -> AbLang2 metrics up to date. Skipping model.")

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
                wt_aa, pos, mut_aa = m[0], int(m[1:-1]) - 1, m[-1]
                masked_seq = wt_list.copy()
                masked_seq[pos] = "[MASK]" 
                
                inputs = pb_tokenizer(" ".join(masked_seq), return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = pb_model(**inputs).logits
                    log_probs = torch.log_softmax(logits[0, pos + 1], dim=-1)
                    mut_delta_logp += (log_probs[pb_tokenizer.convert_tokens_to_ids(mut_aa)] - 
                                       log_probs[pb_tokenizer.convert_tokens_to_ids(wt_aa)]).item()
            df.at[idx, 'protbert_fitness'] = mut_delta_logp

        del pb_model, pb_tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(" -> ProtBERT metrics up to date. Skipping model.")

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
                    wt_aa, pos, mut_aa = m[0], int(m[1:-1]) - 1, m[-1]
                    masked_seq = wt_list.copy()
                    masked_seq[pos] = "<mask>"
                    
                    _, _, mask_tokens = batch_converter_esm([("seq", "".join(masked_seq))])
                    with torch.no_grad():
                        log_probs = torch.log_softmax(model_esm(mask_tokens.to(device))["logits"][0, pos + 1], dim=-1)
                        mut_delta_logp += (log_probs[alphabet_esm.tok_to_idx[mut_aa]] - log_probs[alphabet_esm.tok_to_idx[wt_aa]]).item()
                df.at[idx, 'esm_fitness'] = mut_delta_logp
                
        del model_esm, bindpred_model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(" -> ESM-2 & BINDPRED metrics up to date. Skipping models.")
    
    # ---------------------------------------------------------
    # 10. PLM 5: POET-2 (OpenProtein.AI Evolutionary Fitness)
    # ---------------------------------------------------------
    if 'poet2_fitness' not in df.columns: df['poet2_fitness'] = np.nan
    mask = df['poet2_fitness'].isna()
    
    if mask.any():
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
                    seqs.append(row['VH'] + row['VL'])
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
    # 11. JanusDDG Scoring
    # ---------------------------------------------------------
    if 'janus_ddg' not in df.columns: df['janus_ddg'] = np.nan
    mask = df['janus_ddg'].isna()
    
    if mask.any():
        print(f"Running JanusDDG for {mask.sum()} missing variants...")
        janus_src_path = "/home/esigalas/workspace/JanusDDG/src" 
        if janus_src_path not in sys.path:
            sys.path.append(janus_src_path)
        import utils
        utils.MODELS_DIR = "/home/esigalas/workspace/JanusDDG/models" 
        
        temp_csv_path = os.path.join(run_dir, "temp_janus_input.csv")
        subset_to_compute = df[mask & (df['mutations'] != "WT")]
        
        if len(subset_to_compute) > 0:
            pd.DataFrame({'ID': subset_to_compute['id'], 'Sequence': clean_wt, 
                          'MTS': subset_to_compute['mutations'].str.replace(",", "_")}).to_csv(temp_csv_path, index=False)
            
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
    else:
        print(" -> JanusDDG metrics up to date. Skipping model.")

    # ---------------------------------------------------------
    # 12. Adaptive Normalization & Global Composite Score
    # ---------------------------------------------------------
    def norm(col, invert=False):
        if col not in df.columns or df[col].isna().all(): return 0.5 
        c_min, c_max = df[col].min(), df[col].max()
        if c_max == c_min: return 0.5
        return (c_max - df[col]) / (c_max - c_min) if invert else (df[col] - c_min) / (c_max - c_min)

    df['pi_delta'] = (df['pI'] - 7.4).abs()

    df['camsol_norm'] = norm('camsol_score')
    df['janus_norm'] = norm('janus_ddg', invert=True)      
    df['pyrosetta_norm'] = norm('pyrosetta_ddg', invert=True)  
    df['bindpred_norm'] = norm('bindpred_kd', invert=True) 
    df['esm_norm'] = norm('esm_fitness')
    df['ab_norm'] = norm('antiberty_fitness')
    df['ablang_norm'] = norm('ablang_fitness')
    df['protbert_norm'] = norm('protbert_fitness')
    df['poet2_norm'] = norm('poet2_fitness') 
    df['gravy_norm'] = norm('gravy', invert=True)                 
    df['charge_asym_norm'] = norm('charge_asymmetry', invert=True) 
    df['pi_norm'] = norm('pi_delta', invert=False)                 

    # --- DYNAMIC COMPOSITE CALCULATION ---
    # Automatically find all columns ending in '_norm'
    norm_columns = [c for c in df.columns if c.endswith('_norm')]
    
    # Calculate the mean across however many pillars exist
    df['global_composite_score'] = df[norm_columns].mean(axis=1)
    
    df = df.sort_values('global_composite_score', ascending=False).reset_index(drop=True)

    # --- ENFORCE LOGICAL COLUMN GROUPING FOR BACKEND CACHE ---
    ordered_cols = [
        'id', 'mutations', 'global_composite_score', 
        'janus_ddg', 'pyrosetta_ddg'
    ]
    if df['camsol_score'].notna().any():
        ordered_cols.append('camsol_score')
        
    ordered_cols.extend([
        'pI', 'gravy', 'net_charge', 'charge_asymmetry',
        'bindpred_kd', 'esm_fitness', 'antiberty_fitness', 'ablang_fitness',
        'protbert_fitness', 'poet2_fitness',
        'VH', 'VL'
    ])
    
    df[ordered_cols].to_csv(final_csv_path, index=False)
    
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return df

def run_scfv_developability_pipeline(vh_seq: str, vl_seq: str, base_output_dir: str, run_name: str, 
                                     vh_len: int, vl_len: int, 
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
        print(f" -> Running AntiFold at Temperature {temp}...")
        
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
            subprocess.run(antifold_cmd, check=True, capture_output=True, text=True)
            
            # AntiFold automatically names its output based on the input PDB and chains
            af_fasta = os.path.join(run_dir, f"{run_name}_relaxed_AB.fasta")
            
            if os.path.exists(af_fasta):
                af_records = list(SeqIO.parse(af_fasta, "fasta"))
                
                # Save a clean copy
                SeqIO.write(af_records, os.path.join(run_dir, f"{run_name}_antifold.fasta"), "fasta")
                all_raw_records.extend(af_records)
                
                # FIX: Delete the messy original AntiFold output so it doesn't cause file confusion later!
                os.remove(af_fasta)

        except subprocess.CalledProcessError as e:
            st.error(f"**AntiFold Crashed!**\n\nError Log:\n```\n{e.stderr}\n```")
            raise RuntimeError("Phase 1 halted due to AntiFold failure.")

    # --- DYNAMIC CDR MASKING FOR MPNN MODELS ---
    # FIX 1: Change file extension to .jsonl
    fixed_json_path = os.path.join(run_dir, "fixed_positions.jsonl")
    
    if ("ProteinMPNN (Diverse)" in gen_models) or ("ThermoMPNN (ddG-Driven)" in gen_models) or ("ProteinMPNN (Soluble)" in gen_models) or ("ESM-C (Evolutionary LM)" in gen_models) or ("AbLang2 (Antibody LM)" in gen_models):
        from abnumber import Chain
        
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
    
    metadata_path = os.path.join(run_dir, f"{run_name}_metadata.json")
    run_metadata = {
        "vh_len": vh_len,
        "vl_len": vl_len
    }

    with open(metadata_path, "w") as f:
        json.dump(run_metadata, f)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
