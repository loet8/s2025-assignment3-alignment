#!/usr/bin/env python3
import argparse, glob, json, os, time, random
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_mmlu_from_dir(dev_dir):
    examples = []
    for csv_path in sorted(glob.glob(os.path.join(dev_dir, "*_dev.csv"))):
        df = pd.read_csv(csv_path, header=None)

        if df.shape[1] != 6:
            raise ValueError(f"{csv_path} has {df.shape[1]} columns, expected 6")

        df.columns = ["question", "A", "B", "C", "D", "answer"]

        subject = os.path.basename(csv_path).split("_dev.csv")[0]
        for _, row in df.iterrows():
            examples.append({
                "subject": subject,
                "question": row["question"],
                "options": [row[o] for o in ["A","B","C","D"]],
                "answer":  row["answer"]
            })
    return examples

def format_prompt(ex):
    opts = "\n".join(f"{chr(65+i)}. {opt}" for i,opt in enumerate(ex["options"]))
    return (
        f"Answer the following multiple choice question about {ex['subject']}."
        " Respond with a single sentence of the form \"The correct answer is _\","
        " filling the blank with the letter corresponding to the correct answer (i.e., A, B, C or D).\n\n"
        f"Question: {ex['question']}\n{opts}\nAnswer:"
    )

def extract_prediction(text):
    for ch in text.strip():
        if ch in "ABCD":
            return ch
    return ""

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--mmlu-dir",  required=True)
    p.add_argument("--device",    default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device)
    model.eval()

    examples = load_mmlu_from_dir(args.mmlu_dir)
    results = []
    t0 = time.time()
    for ex in examples:
        prompt = format_prompt(ex)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        pred = extract_prediction(gen)
        results.append({
            "question": ex["question"],
            "predicted": pred,
            "correct": ex["answer"],
            "is_correct": pred == ex["answer"],
            "model_output": gen.strip(),
        })
    elapsed = time.time() - t0

    total   = len(results)
    correct = sum(r["is_correct"] for r in results)
    print(f"→ Throughput: {total/elapsed:.2f} examples/s")
    print(f"→ Accuracy:   {correct/total*100:.2f}% ({correct}/{total})")

    errors = [r for r in results if not r["is_correct"]]
    print("\nSample errors:")
    for r in random.sample(errors, min(10,len(errors))):
        print(f"Q: {r['question']}\n  → out: {r['model_output']}  pred: {r['predicted']}  gold: {r['correct']}\n")

    with open("mmlu_sft_results.json","w") as f:
        json.dump(results, f, indent=2)
    print("Saved full results to mmlu_sft_results.json")

if __name__=="__main__":
    main()
