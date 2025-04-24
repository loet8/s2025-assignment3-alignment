import argparse
import json
import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_sst(path):
    df = pd.read_csv(path)
    if   "prompt"        in df.columns: prompt_col = "prompt"
    elif "prompts_final" in df.columns: prompt_col = "prompts_final"
    else:
        raise ValueError(f"No `prompt` or `prompts_final` column in {path}")
    examples = []
    for _, row in df.iterrows():
        examples.append({
            "id":     row["id"],
            "prompt": row[prompt_col]
        })
    return examples

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir",     required=True)
    p.add_argument("--tokenizer-dir", help="defaults to model-dir")
    p.add_argument("--sst-csv",       required=True)
    p.add_argument("--device",        default="cuda")
    args = p.parse_args()

    device    = torch.device(args.device)
    tok_dir   = args.tokenizer_dir or args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
                    args.model_dir,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=True,
                ).to(device)
    model.eval()

    examples = load_sst(args.sst_csv)
    out_path = "sst_sft_preds.jsonl"
    fout     = open(out_path, "w")
    t0       = time.time()

    for ex in examples:
        inputs = tokenizer(ex["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        resp = tokenizer.decode(
            gen_ids[0, inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        ).strip()

        fout.write(json.dumps({
            "id":             ex["id"],
            "prompt":         ex["prompt"],
            "model_response": resp
        }) + "\n")

    fout.close()
    elapsed   = time.time() - t0
    throughput= len(examples) / elapsed
    print(f"SST throughput: {throughput:.2f} examples/s")
    print(f"Wrote {len(examples)} predictions to {out_path}")

if __name__=="__main__":
    main()
