import os
import sys
import pandas as pd
import numpy as np
from Bio import SeqIO
import seaborn as sns
import matplotlib.pyplot as plt
import torch
    
import subprocess
import shutil
import urllib.request
from transformers import AutoTokenizer, EsmForProteinFolding
import pyrosetta
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
import antifold
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from catboost import CatBoostRegressor
from huggingface_hub import hf_hub_download
import gc

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
    import esm
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
    from transformers import AutoModelForMaskedLM, AutoTokenizer
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
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    
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

def run_scfv_developability_pipeline(vh_seq: str, vl_seq: str, base_output_dir: str, run_name: str):
    
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
                    new_res_idx = res_idx - vh_len
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
    # STEP 4: Variant Generation & CDR Masking (AntiFold)
    # ---------------------------------------------------------    
    print("Step 4: Running AntiFold Inverse Folding...")

    antifold_code_dir = os.path.dirname(antifold.__file__)
    site_packages_dir = os.path.dirname(antifold_code_dir)
    buggy_expected_model_path = os.path.join(site_packages_dir, "models", "model.pt")

    manual_model_path = os.path.join(antifold_code_dir, "models", "model.pt")
    local_model_path = os.path.join(os.getcwd(), "models", "model.pt")

    if not os.path.exists(buggy_expected_model_path):
        os.makedirs(os.path.dirname(buggy_expected_model_path), exist_ok=True)
        if os.path.exists(manual_model_path):
            shutil.copy(manual_model_path, buggy_expected_model_path)
            print(f"Fixed bug: Copied weights to {buggy_expected_model_path}")
        elif os.path.exists(local_model_path):
            shutil.copy(local_model_path, buggy_expected_model_path)
            print(f"Fixed bug: Copied weights to {buggy_expected_model_path}")
        else:
            print("ERROR: Could not find the model.pt file you downloaded earlier!")
        
    antifold_cmd = [
        "python", "-m", "antifold.main",
        "--pdb_file", relaxed_pdb_path,
        "--heavy_chain", "A",
        "--light_chain", "B",
        "--regions", "FWH FWL",
        "--num_seq_per_target", "100",
        "--sampling_temp", "0.2",
        "--out_dir", run_dir
    ]
    
    try:
        subprocess.run(antifold_cmd, check=True)
        print(f"AntiFold run finished. Raw variants saved in {run_dir}")
    except subprocess.CalledProcessError as e:
        print(f"AntiFold encountered an error: {e}")
        return

    # ---------------------------------------------------------
    # STEP 5: Post-Processing Clean FASTA for CamSol Web Server
    # ---------------------------------------------------------
    raw_fasta_path = os.path.join(run_dir, f"{run_name}_relaxed_AB.fasta")
    camsol_fasta_path = os.path.join(run_dir, f"{run_name}_camsol_upload.fasta")

    if os.path.exists(raw_fasta_path):
        print(f"Step 5: Formatting clean FASTA for CamSol server -> {camsol_fasta_path}")
        cleaned_records = []
        
        for i, record in enumerate(SeqIO.parse(raw_fasta_path, "fasta")):
            # Strip non-standard chain breaks (: and /)
            clean_seq = str(record.seq).replace(":", "").replace("/", "")
            record.seq = record.seq.__class__(clean_seq)
            
            # Assign unique IDs for the web server to prevent pandas merge explosions
            record.id = f"{run_name}_WT" if i == 0 else f"{run_name}_sample_{i}"
            record.description = ""
            
            cleaned_records.append(record)
            
        # Only save the CamSol payload. Do NOT overwrite the raw FASTA.
        SeqIO.write(cleaned_records, camsol_fasta_path, "fasta")
        print(f"Step 5 Complete: {camsol_fasta_path} ready for web upload.")
    # ---------------------------------------------------------
    # Cleanup: Free GPU Memory
    # ---------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
