#!/usr/bin/env python3
import argparse, os, json, time, torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_mmlu_from_dir(path):
    examples = []
    for fn in os.listdir(path):
        if not fn.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(path, fn))
        # find the question and answer columns
        qcols = [c for c in df.columns if "question" in c.lower()]
        acols = [c for c in df.columns if c.lower() == "answer"]
        if not qcols or not acols:
            raise ValueError(f"Could not find `question`/`answer` cols in {fn}: {df.columns}")
        qcol, acol = qcols[0], acols[0]
        for _, row in df.iterrows():
            examples.append({
                "question": str(row[qcol]).strip(),
                "correct_answer": str(row[acol]).strip()
            })
    return examples

def format_prompt(question):
    return (
        "Answer the following multiple choice question.  "
        "Respond with a single sentence of the form \"The correct answer is _\".\n\n"
        f"Question: {question}\n"
        "A. ...\nB. ...\nC. ...\nD. ...\n"
        "Answer:"
    )

def extract_letter(text):
    for tok in text.strip().split():
        tok = tok.strip(".,\"'")
        if len(tok)==1 and tok.upper() in list("ABCDE"):
            return tok.upper()
    return ""

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--tokenizer-dir", help="default: same as model-dir")
    p.add_argument("--mmlu-dir",   required=True)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--out-file",   default="mmlu_sft_preds.jsonl")
    args = p.parse_args()

    device   = torch.device(args.device)
    tok_dir  = args.tokenizer_dir or args.model_dir
    tokenizer= AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    model    = AutoModelForCausalLM.from_pretrained(
                   args.model_dir,
                   torch_dtype=torch.bfloat16,
                   attn_implementation="flash_attention_2",
                   trust_remote_code=True
               ).to(device)
    model.eval()

    examples = load_mmlu_from_dir(args.mmlu_dir)
    total = correct = 0
    outputs = []
    t0 = time.time()

    for ex in examples:
        prompt = format_prompt(ex["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        out_text = tokenizer.decode(
            gen[0, inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        ).strip()
        pred = extract_letter(out_text)
        is_correct = (pred == ex["correct_answer"])
        total += 1
        correct += is_correct

        outputs.append({
            "question":       ex["question"],
            "prompt":         prompt,
            "model_output":   out_text,
            "predicted":      pred,
            "correct_answer": ex["correct_answer"],
            "is_correct":     is_correct,
        })

    elapsed = time.time() - t0
    throughput = total / elapsed
    accuracy   = correct / total * 100

    print(f"Throughput: {throughput:.2f} examples/s")
    print(f"Accuracy:   {accuracy:.2f}% ({correct}/{total})")

    with open(args.out_file, "w") as f:
        for entry in outputs:
            f.write(json.dumps(entry) + "\n")
    print(f"Wrote {len(outputs)} predictions to {args.out_file}")

if __name__=="__main__":
    main()
