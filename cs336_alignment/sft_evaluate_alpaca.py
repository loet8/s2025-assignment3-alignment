import os
import json
import time
import argparse
from vllm import LLM, SamplingParams


def load_alpaca_eval_examples(filepath: str):
    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            examples.append(ex)
    return examples


def format_prompt(ex: dict) -> str:
    instr = ex.get('instruction', '')
    inp   = ex.get('input', '')
    if inp:
        return f"{instr}\n{inp}"
    return instr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True,
                        help="Path to your fine-tuned Qwen2.5-0.5B model")
    parser.add_argument("--ae-path",    required=True,
                        help="Path to AlpacaEval JSONL file (e.g. alpaca_eval.jsonl)")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--out-file",   default="alpaca_sft_outputs.json")
    args = parser.parse_args()

    examples = load_alpaca_eval_examples(args.ae_path)
    print(f"Loaded {len(examples)} AlpacaEval examples")

    llm = LLM(model=args.model_dir)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop=["\n"],
    )

    prompts = [format_prompt(ex) for ex in examples]

    t0 = time.time()
    outputs = []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        batch_outputs = llm.generate(batch, sampling_params)
        outputs.extend(batch_outputs)
    elapsed = time.time() - t0
    throughput = len(prompts) / elapsed

    preds = []
    for ex, out in zip(examples, outputs):
        text = out.outputs[0].text
        preds.append({
            "instruction": ex.get("instruction", ""),
            "output":      text,
            "generator":   "qwen2.5-0.5b-sft",
            "dataset":     ex.get("dataset", "alpaca_eval"),
        })

    with open(args.out_file, 'w', encoding='utf-8') as fout:
        json.dump(preds, fout, ensure_ascii=False)
    print(f"Wrote {len(preds)} SFT predictions to {args.out_file}")
    print(f"Throughput: {throughput:.2f} ex/s")

if __name__ == '__main__':
    main()
