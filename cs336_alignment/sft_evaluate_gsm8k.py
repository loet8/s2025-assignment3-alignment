import os
import json
import time
import argparse
import torch
from typing import Any, Dict, List
from vllm import LLM, SamplingParams
from tests import adapters 

def load_gsm8k_examples(filepath: str) -> List[Dict[str, Any]]:
    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            examples.append({
                "question": ex["question"].strip(),
                "answer": ex.get("parsed_answer") or ex.get("answer").split()[-1]
            })
    return examples

def format_prompt(example: Dict[str, Any]) -> str:
    return f"{example['question']}\n\nAnswer:"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir",    required=True,
                        help="Path to fine-tuned Qwen2.5-0.5B model")
    parser.add_argument("--gsm8k-path",   required=True,
                        help="Path to the GSM8K JSONL file")
    parser.add_argument("--batch-size",   type=int, default=5)
    parser.add_argument("--max-tokens",   type=int, default=1024)
    parser.add_argument("--out-file",     default="gsm8k_sft_results.json")
    args = parser.parse_args()

    examples = load_gsm8k_examples(args.gsm8k_path)
    print(f"Loaded {len(examples)} examples for GSM8K SFT evaluation")

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
        outputs.extend(llm.generate(batch, sampling_params))
        torch.cuda.empty_cache()
    elapsed = time.time() - t0

    num_correct = 0
    results = []
    for ex, prompt, out in zip(examples, prompts, outputs):
        text = out.outputs[0].text
        pred = adapters.run_parse_gsm8k_response(text)
        corr = ex["answer"]
        is_corr = (pred == corr)
        if is_corr:
            num_correct += 1
        results.append({
            "question":       ex["question"],
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
            "throughput":   throughput,
            "accuracy":     accuracy,
            "num_correct":  num_correct,
            "total":        len(prompts),
            "results":      results,
        }, fout, indent=2)
    print(f"Wrote detailed results to {args.out_file}")

if __name__ == "__main__":
    main()
