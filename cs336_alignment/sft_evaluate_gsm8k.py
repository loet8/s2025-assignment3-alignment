#!/usr/bin/env python3
import argparse
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_gsm8k(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            ex = json.loads(line)
            data.append({
                "question": ex["question"],
                "answer": ex.get("parsed_answer") or ex.get("answer").split()[-1],
            })
    return data

def format_prompt(question):
    return f"{question}\n\nAnswer:"

def extract_number(text):
    for token in text.strip().split():
        token = token.strip(".,")
        if token.replace(".", "", 1).isdigit():
            return token
    return ""

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir",     required=True)
    p.add_argument("--tokenizer-dir", help="Defaults to model-dir")
    p.add_argument("--gsm8k-path",    required=True)
    p.add_argument("--device",        default="cuda")
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

    examples = load_gsm8k(args.gsm8k_path)
    total, correct = 0, 0
    t0 = time.time()

    for ex in examples:
        prompt = format_prompt(ex["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        gen = tokenizer.decode(out[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        pred = extract_number(gen)
        if pred == ex["answer"]:
            correct += 1
        total += 1

    elapsed = time.time() - t0
    throughput = total / elapsed
    accuracy = correct / total * 100

    print(f"Throughput: {throughput:.2f} examples/s")
    print(f"Accuracy:   {accuracy:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    main()
