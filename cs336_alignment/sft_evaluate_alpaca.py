import os
import json
import time
import sys

repo_root = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, repo_root)

from vllm import LLM, SamplingParams
from tests import adapters  # required if using custom tokenizer logic


model_dir = "/content/models/qwen2.5-0.5B-sft"
alpaca_eval_path = "/content/data/alpaca_eval/alpaca_eval.jsonl"
batch_size = 5
max_tokens = 1024
out_file = "alpaca_eval_sft_outputs.jsonl"


def load_alpaca_eval_examples(filepath: str):
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line.strip()))
    return examples


def format_instruction_prompt(example):
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        "### Response:"
    )


def main():
    examples = load_alpaca_eval_examples(alpaca_eval_path)
    prompts = [format_instruction_prompt(ex) for ex in examples]

    model = LLM(model=model_dir)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens, stop=["###"])

    print(f"Evaluating {len(prompts)} AlpacaEval examples...")
    start_time = time.time()
    outputs = model.generate(prompts, sampling_params=sampling_params, batch_size=batch_size)
    end_time = time.time()

    with open(out_file, "w", encoding="utf-8") as f:
        for ex, out, prompt in zip(examples, outputs, prompts):
            response = out.outputs[0].text.strip()
            record = {
                "instruction": ex["instruction"],
                "response": response,
                "prompt": prompt
            }
            f.write(json.dumps(record) + "\n")

    throughput = len(prompts) / (end_time - start_time)
    print(f"Throughput: {throughput:.2f} examples/second")
    print(f"Results saved to: {out_file}")

if __name__ == "__main__":
    main()
