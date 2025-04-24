import argparse
import json
import os
import sys
import time
from pathlib import Path

repo_root = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, repo_root)


from vllm import LLM, SamplingParams
from tests import adapters  

def format_sft_prompt(example: dict) -> str:
    """Wraps an MMLU example in the Alpaca-style instruction-tuning template."""
    q       = example["question"]
    opts    = example["options"]
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{q}\n"
        f"A. {opts[0]}\n"
        f"B. {opts[1]}\n"
        f"C. {opts[2]}\n"
        f"D. {opts[3]}\n\n"
        "### Response:"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",
                        help="Path to your fine-tuned Qwen2.5-0.5B model")
    parser.add_argument("--mmlu-dir",
                        help="Directory with MMLU JSONL files")
    parser.add_argument("--batch-size", type=int, default=5)     
    parser.add_argument("--max-tokens",   type=int, default=512)
               
    parser.add_argument("--out-file", default="mmlu_sft_results.json",
                        help="Where to write evaluation results")
    args = parser.parse_args()

    examples = []
    for fn in Path(args.mmlu_dir).glob("*.jsonl"):
        with open(fn) as f:
            for line in f:
                examples.append(json.loads(line))

    prompts = [format_sft_prompt(ex) for ex in examples]

    llm = LLM(model=args.model_path)
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=512, stop=["\n"]
    )

    results = []
    t_start = time.time()
    for i in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[i : i + args.batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        for ex, out in zip(examples[i : i + args.batch_size], outputs):
            gen = out.outputs[0].text.strip()
            pred = adapters.parse_mmlu_response(gen)
            corr = ex["answer"]
            results.append({
                "example": ex,
                "prompt": batch_prompts[0],
                "model_output": gen,
                "predicted": pred,
                "correct_answer": corr,
                "is_correct": (pred == corr)
            })

    elapsed = time.time() - t_start
    throughput = len(prompts) / elapsed
    accuracy   = sum(r["is_correct"] for r in results) / len(results) * 100

    out = {
        "throughput": throughput,
        "accuracy": accuracy,
        "total": len(results),
        "results": results
    }
    with open(args.out_file, "w") as fout:
        json.dump(out, fout, indent=2)

if __name__ == "__main__":
    main()
