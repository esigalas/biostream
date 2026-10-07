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

def score_scfv_variants(run_dir: str, run_name: str, wt_sequence: str):
    print(f"--- Phase 2: Scoring Variants for {run_name} ---")
    
    # ---------------------------------------------------------
    # 1. File Checks & Path Injection
    # ---------------------------------------------------------
    raw_fasta_path = os.path.join(run_dir, f"{run_name}_relaxed_AB.fasta")
    camsol_txt_path = os.path.join(run_dir, f"{run_name}_camsol.txt")
    
    if not os.path.exists(raw_fasta_path):
        raise FileNotFoundError(f"Cannot find {raw_fasta_path}. Run Phase 1 first.")
    if not os.path.exists(camsol_txt_path):
        raise FileNotFoundError(f"Cannot find {camsol_txt_path}. Upload your FASTA to the CamSol web server, download the results, and save it here.")

    # Inject the JanusDDG src directory directly into Python's path
    janus_src_path = "/home/esigalas/workspace/JanusDDG/src" 
    if janus_src_path not in sys.path:
        sys.path.append(janus_src_path)
        
    try:
        import utils
        import esm
        # Hijack the internal MODELS_DIR variable so utils.load_model 
        # looks in your cloned repo instead of the current working directory
        utils.MODELS_DIR = "/home/esigalas/workspace/JanusDDG/models" 
    except ImportError as e:
        raise ImportError(f"Failed to load Janus dependencies. Ensure 'fair-esm' is installed and path is correct. ({e})")

    # ---------------------------------------------------------
    # 2. Parse FASTA and Extract Mutations & Chains
    # ---------------------------------------------------------
    records = list(SeqIO.parse(raw_fasta_path, "fasta"))
    variants = []
    
    clean_wt = wt_sequence.replace(":", "")
    
    for i, record in enumerate(records):
        # 1. Generate the exact same ID used in the CamSol upload to allow pandas merging
        unique_id = f"{run_name}_WT" if i == 0 else f"{run_name}_sample_{i}"
        
        # 2. Use the native AntiFold delimiter to isolate the chains
        raw_seq_str = str(record.seq)
        if "/" in raw_seq_str:
            vh, vl = raw_seq_str.split("/")
        elif ":" in raw_seq_str:
            vh, vl = raw_seq_str.split(":")
        else:
            vh, vl = raw_seq_str, ""
            
        mut_seq = vh + vl
        mutations = []
        
        # 3. Calculate mutations
        if mut_seq == clean_wt:
            mut_string = "WT"
        else:
            for idx, (wt_aa, mut_aa) in enumerate(zip(clean_wt, mut_seq)):
                if wt_aa != mut_aa:
                    mutations.append(f"{wt_aa}{idx+1}{mut_aa}")
            mut_string = ",".join(mutations)
            
        variants.append({
            "id": unique_id,
            "mutations": mut_string,
            "VH": vh,  # Save independent chains
            "VL": vl
        })
        
    df = pd.DataFrame(variants)
    print(f"Loaded {len(df)} variants. Identifying mutations...")
    
    # ---------------------------------------------------------
    # 3. Integrate Web Server CamSol Scores
    # ---------------------------------------------------------
    print("Merging CamSol web server predictions...")
    camsol_df = pd.read_csv(camsol_txt_path, sep='\t')
    
    if len(camsol_df) != len(df):
        raise ValueError(f"Mismatch: CamSol file has {len(camsol_df)} rows, but FASTA has {len(df)}.")
        
    df['camsol_score'] = camsol_df['protein variant score'].values
    
    # ---------------------------------------------------------
    # 4. Global Model Initialization (ESM-2 & BINDPRED)
    # ---------------------------------------------------------
    print("Initializing ESM-2 Model for BINDPRED and JanusDDG...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_esm, alphabet_esm = esm.pretrained.esm2_t33_650M_UR50D()
    model_esm = model_esm.to(device)
    batch_converter_esm = alphabet_esm.get_batch_converter()
    model_esm.eval()

    print("Downloading and Loading BINDPRED CatBoost Model...")
    model_path = hf_hub_download(repo_id="hbp5181/BindPred", filename="ESM2_BindPred.cbm")
    bindpred_model = CatBoostRegressor()
    bindpred_model.load_model(model_path, format="cbm")

    # ---------------------------------------------------------
    # 5. Calculate Net Charge & BINDPRED Inference
    # ---------------------------------------------------------
    print("Calculating Net Charge and running BINDPRED inference...")
    bindpred_scores = []
    net_charges = []
    
    for _, row in df.iterrows():
        # 1. Net Charge at pH 7.4
        full_seq = row['VH'] + row['VL']
        charge = ProteinAnalysis(full_seq).charge_at_pH(7.4)
        net_charges.append(charge)
        
        # 2. Extract ESM-2 Embeddings
        data = [("VH", row['VH']), ("VL", row['VL'])]
        _, _, batch_tokens = batch_converter_esm(data)
        batch_tokens = batch_tokens.to(device)
        
        with torch.no_grad():
            results = model_esm(batch_tokens, repr_layers=[33], return_contacts=False)
            token_reps = results["representations"][33]
            
            # Extract mean-pooled tensors and move them back to CPU/Numpy for CatBoost
            vh_emb = token_reps[0, 1 : len(row['VH']) + 1].mean(0).cpu().numpy()
            vl_emb = token_reps[1, 1 : len(row['VL']) + 1].mean(0).cpu().numpy()
            
            # 3. Stack features (1280 + 1280 = 2560 dimensions) and predict
            combined_features = np.hstack([vh_emb, vl_emb])
            kd_score = bindpred_model.predict(combined_features)
            
            bindpred_scores.append(kd_score)

    df['net_charge'] = net_charges
    df['bindpred_kd'] = bindpred_scores

    
    # ---------------------------------------------------------
    # 6. JanusDDG Scoring (Thermodynamic Stability)
    # ---------------------------------------------------------
    print("Running JanusDDG stability predictions locally...")
    
    # Isolate only mutated sequences for Janus
    janus_df = df[df['mutations'] != "WT"].copy()
    
    # Format the temporary CSV exactly as utils.py expects
    janus_input = pd.DataFrame({
        'ID': janus_df['id'],
        'Sequence': clean_wt, 
        'MTS': janus_df['mutations'].str.replace(",", "_") 
    })
    
    temp_csv_path = os.path.join(run_dir, "temp_janus_input.csv")
    janus_input.to_csv(temp_csv_path, index=False)
    
    # --- HOTFIX FOR PYTORCH UNPICKLER ---
    import __main__
    import model as janus_model
    
    # 1. Patch the missing global device variable in their model.py file
    janus_model.device = device
    
    # 2. Map their architectures into the notebook's global scope so torch.load can find them
    setattr(__main__, 'Cross_Attention_DDG', janus_model.Cross_Attention_DDG)
    setattr(__main__, 'TransformerRegression', janus_model.TransformerRegression)
    setattr(__main__, 'SinusoidalPositionalEncoding', janus_model.SinusoidalPositionalEncoding)
    # ------------------------------------

    model_janus = utils.load_model('JanusDDG_fine_tuned.pth', device)
    
    # Execute the academic pipeline
    pred_dir, _ = utils.process_and_predict(temp_csv_path, model_janus, model_esm, batch_converter_esm, device)
    
    # Map predictions back to the main dataframe (filling WT with 0.0)
    janus_df['janus_ddg'] = pred_dir.values
    df = df.merge(janus_df[['id', 'janus_ddg']], on='id', how='left')
    df['janus_ddg'] = df['janus_ddg'].fillna(0.0)
    
    # Clean up temporary file
    os.remove(temp_csv_path)

    # ---------------------------------------------------------
    # 5. Evaluation Metrics & Global Composite Score
    # ---------------------------------------------------------
    print("Establishing survival thresholds and global composite scores...")
    
    # 1. Baseline Survival Status (Required for Step 6 Seaborn diagnostic plot)
    camsol_10th_p = df['camsol_score'].quantile(0.10)
    df['survives_ddg'] = df['janus_ddg'] >= -1.0
    df['survives_camsol'] = df['camsol_score'] >= camsol_10th_p
    df['survival_status'] = df['survives_ddg'] & df['survives_camsol']
    
    # 2. Normalize CamSol (Higher is better -> 1.0)
    c_min, c_max = df['camsol_score'].min(), df['camsol_score'].max()
    camsol_norm = (df['camsol_score'] - c_min) / (c_max - c_min) if c_max > c_min else 0.5
    
    # 3. Normalize JanusDDG (Lower/more negative is better -> 1.0)
    j_min, j_max = df['janus_ddg'].min(), df['janus_ddg'].max()
    janus_norm = (j_max - df['janus_ddg']) / (j_max - j_min) if j_max > j_min else 0.5
    
    # 4. Normalize BINDPRED (Lower/more negative Kd is better -> 1.0)
    b_min, b_max = df['bindpred_kd'].min(), df['bindpred_kd'].max()
    bindpred_norm = (b_max - df['bindpred_kd']) / (b_max - b_min) if b_max > b_min else 0.5
    
    # 5. Balanced Composite Score (0.0 = worst, 1.0 = best across batch)
    df['global_composite_score'] = (camsol_norm + janus_norm + bindpred_norm) / 3.0
    df = df.sort_values('global_composite_score', ascending=False).reset_index(drop=True)

    # ---------------------------------------------------------
    # 6. Diagnostic Visualization
    # ---------------------------------------------------------
    output_csv = os.path.join(run_dir, f"{run_name}_developability_metrics.csv")
    df.to_csv(output_csv, index=False)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df, x='janus_ddg', y='camsol_score', 
        hue='survival_status', palette={True: "#2ca02c", False: "#d62728"},
        s=70, edgecolor="w", alpha=0.8
    )
    plt.axvline(-1.0, color='gray', linestyle='--', label="DDG Threshold")
    plt.axhline(camsol_10th_p, color='gray', linestyle=':', label="CamSol 10th %ile")
    plt.title(f"Variant Survival: {df['survival_status'].sum()} / {len(df)} Passed")
    plt.xlabel("JanusDDG (ΔΔG)")
    plt.ylabel("CamSol Intrinsic Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plot_path = os.path.join(run_dir, f"{run_name}_diagnostic_scatter.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Phase 2 complete! Results ready at {output_csv}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
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
