import os
import csv
import json
import torch
import argparse
import time
from typing import Any, List, Dict
from vllm import LLM, SamplingParams
from tests import adapters

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def load_mmlu_csv(csv_path: str, subject: str) -> List[Dict[str, Any]]:
    examples = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 6:
                continue
            example = {
                "subject": subject,
                "question": row[0],
                "options": [row[1], row[2], row[3], row[4]],
                "answer": row[-1].strip().upper()
            }
            examples.append(example)
    return examples

def load_all_examples(base_dir: str) -> List[Dict[str, Any]]:
    examples = []
    for file in os.listdir(base_dir):
        if file.endswith(".csv"):
            base_name = os.path.splitext(file)[0]
            subject = base_name.replace("_dev", "").replace("_test", "").replace("_val", "")
            csv_path = os.path.join(base_dir, file)
            print(f"Loading {csv_path} for subject: {subject}")
            examples.extend(load_mmlu_csv(csv_path, subject))
    return examples

def format_prompt(example: Dict[str, Any]) -> str:
    prompt = (
        f"Answer the following multiple choice question about {example['subject']}. "
        f"Respond with a single sentence of the form \"The correct answer is _\", "
        f"filling the blank with the letter corresponding to the correct answer (i.e., A, B, C or D).\n\n"
        f"Question: {example['question']}\n"
        f"A. {example['options'][0]}\n"
        f"B. {example['options'][1]}\n"
        f"C. {example['options'][2]}\n"
        f"D. {example['options'][3]}\n"
        f"Answer:"
    )
    return prompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True,
                   help="Path to your fine-tuned Qwen2.5-0.5B model")
    p.add_argument("--mmlu-dir", required=True,
                   help="Directory containing MMLU .csv files (dev/test splits)")
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--out-file", default="mmlu_sft_results.json")
    args = p.parse_args()

    examples = load_all_examples(args.mmlu_dir)
    print(f"Loaded {len(examples)} examples")

    prompts = [format_prompt(ex) for ex in examples]
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop=["\n"],
    )
    llm = LLM(model=args.model_dir)

    t0 = time.time()
    outputs = []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        batch_out = llm.generate(batch, sampling_params)
        outputs.extend(batch_out)
        torch.cuda.empty_cache()
    elapsed = time.time() - t0

    num_correct = 0
    results = []
    for ex, prompt, out in zip(examples, prompts, outputs):
        text = out.outputs[0].text
        pred = adapters.run_parse_mmlu_response(ex, text)
        corr = ex["answer"]
        is_corr = (pred == corr)
        if is_corr:
            num_correct += 1
        results.append({
            "example":        ex,
            "prompt":         prompt,
            "model_output":   text,
            "predicted":      pred,
            "correct_answer": corr,
            "is_correct":     is_corr,
        })

    throughput = len(prompts) / elapsed
    accuracy   = num_correct / len(prompts) * 100

    print(f"Throughput: {throughput:.2f} examples/s")
    print(f"Accuracy:   {accuracy:.2f}% ({num_correct}/{len(prompts)})")

    with open(args.out_file, "w", encoding="utf-8") as fout:
        json.dump({
            "throughput": throughput,
            "accuracy":   accuracy,
            "num_correct": num_correct,
            "total":      len(prompts),
            "results":    results,
        }, fout, indent=2)
    print(f"Results saved to {args.out_file}")

if __name__ == "__main__":
    main()
