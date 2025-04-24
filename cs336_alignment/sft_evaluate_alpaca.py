import argparse
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_alpacaeval(path):
    examples = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            examples.append({
                "id": obj["id"],
                "instruction": obj["instruction"],
                "input": obj.get("input","")
            })
    return examples

def format_prompt(inst, inp):
    if inp:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{inst}

### Input:
{inp}

### Response:"""
    else:
        return f"""Below is an instruction. Write a response that appropriately completes the request.

### Instruction:
{inst}

### Response:"""

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir",    required=True)
    p.add_argument("--tokenizer-dir",help="defaults to model-dir")
    p.add_argument("--ae-path",      required=True,
                   help="AlpacaEval JSONL file")
    p.add_argument("--device",       default="cuda")
    p.add_argument("--out",          default="alpacaeval_sft_preds.json")
    args = p.parse_args()

    device    = torch.device(args.device)
    tok_dir   = args.tokenizer_dir or args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
                    args.model_dir,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=True
                ).to(device)
    model.eval()

    examples = load_alpacaeval(args.ae_path)
    start = time.time()
    outputs = []

    for ex in examples:
        prompt = format_prompt(ex["instruction"], ex["input"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        resp = tokenizer.decode(
            gen[0, inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        ).strip()
        outputs.append({
            "id": ex["id"],
            "prompt": prompt,
            "model_response": resp,
        })

    elapsed = time.time() - start
    throughput = len(outputs) / elapsed
    print(f"→ AlpacaEval throughput: {throughput:.2f} ex/s")

    # save predictions
    with open(args.out, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"Saved {len(outputs)} model outputs to {args.out}")

if __name__=="__main__":
    main()
