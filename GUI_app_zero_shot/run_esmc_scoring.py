# run_esmc_scoring.py
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wt_seq", required=True, type=str)
    parser.add_argument("--input_csv", required=True, type=str)
    parser.add_argument("--output_csv", required=True, type=str)
    args = parser.parse_args()

    clean_wt = args.wt_seq.strip()
    df = pd.read_csv(args.input_csv)

    # 1. Collect all unique 0-indexed positions mutated across the cohort
    unique_positions = set()
    for mut_str in df['mutations'].dropna():
        if mut_str == "WT":
            continue
        for m in str(mut_str).split(","):
            pos = int(m[1:-1]) - 1
            unique_positions.add(pos)

    # 2. Load the exact same Biohub HF checkpoint cached in Phase 1
    hf_id = "biohub/ESMC-600M-hf"
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForMaskedLM.from_pretrained(hf_id).eval()
    mask_token_id = tokenizer.mask_token_id

    if torch.cuda.is_available():
        model = model.cuda()

    inputs = tokenizer(clean_wt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Find the start index after the <cls> token (typically index 1)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    seq_start_idx = 1
    for idx, tok in enumerate(tokens):
        if tok.replace("<", "").replace(">", "").upper() == clean_wt[0].upper():
            seq_start_idx = idx
            break

    # 3. Compute masked-marginal log-probabilities once per unique mutated position
    pos_log_probs = {}
    with torch.no_grad():
        for pos in sorted(unique_positions):
            token_pos = pos + seq_start_idx
            masked_input_ids = inputs["input_ids"].clone()
            masked_input_ids[0, token_pos] = mask_token_id

            logits = model(input_ids=masked_input_ids).logits[0, token_pos]
            pos_log_probs[pos] = F.log_softmax(logits.float(), dim=-1).cpu()

    # 4. Sum the delta log-likelihood for each variant
    scores = []
    for _, row in df.iterrows():
        mut_str = str(row['mutations'])
        if mut_str == "WT" or not mut_str or mut_str == "nan":
            scores.append(0.0)
            continue

        delta_log_p = 0.0
        for m in mut_str.split(","):
            wt_aa, pos, mut_aa = m[0], int(m[1:-1]) - 1, m[-1]
            log_probs = pos_log_probs[pos]
            wt_tok_id = tokenizer.convert_tokens_to_ids(wt_aa)
            mut_tok_id = tokenizer.convert_tokens_to_ids(mut_aa)
            delta_log_p += (log_probs[mut_tok_id] - log_probs[wt_tok_id]).item()

        scores.append(delta_log_p)

    df['esmc_fitness'] = scores
    df[['id', 'esmc_fitness']].to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    main()
