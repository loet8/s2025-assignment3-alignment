import os
import sys
import json
import time
import random

repo_root = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, repo_root)

from vllm import LLM, SamplingParams
from tqdm import tqdm
from typing import Any, List, Dict
from tests import adapters  

model_dir = "/content/models/qwen2.5-0.5B-sft"
gsm8k_path = "/content/data/gsm8k/dev.jsonl"
batch_size = 5
max_tokens = 512
out_file = "gsm8k_sft_outputs.jsonl"


def load_gsm8k(filepath: str) -> List[Dict[str, Any]]:
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


def format_instruction_prompt(example):
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{example['question']}\n\n"
        "### Response:"
    )


def main():
    examples = load_gsm8k(gsm8k_path)
    prompts = [format_instruction_prompt(ex) for ex in examples]
    labels = [ex["answer"].strip() for ex in examples]

    model = LLM(model=model_dir)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens, stop=["\n"])

    print(f"Evaluating {len(prompts)} GSM8K examples...")
    start_time = time.time()
    outputs = []
    for i in tqdm(range(0, len(prompts), 1)):  
        sub_outputs = model.generate(prompts[i:i+1], sampling_params=sampling_params)
        outputs.extend(sub_outputs)
    end_time = time.time()

    correct_count = 0
    results = []

    with open(out_file, "w", encoding="utf-8") as f:
        for ex, out, label, prompt in zip(examples, outputs, labels, prompts):
            raw_response = out.outputs[0].text.strip()
            predicted = raw_response.split()[0].strip().strip(".")  # crude match

            is_correct = label.startswith(predicted)
            correct_count += int(is_correct)

            result = {
                "question": ex["question"],
                "correct": label,
                "predicted": predicted,
                "is_correct": is_correct,
                "model_output": raw_response,
                "prompt": prompt
            }
            results.append(result)
            f.write(json.dumps(result) + "\n")

    total = len(results)
    accuracy = correct_count / total * 100
    throughput = total / (end_time - start_time)

    print(f"\Accuracy: {accuracy:.2f}% ({correct_count}/{total})")
    print(f"Throughput: {throughput:.2f} examples/sec")
    print(f"Results saved to: {out_file}")

if __name__ == "__main__":
    main()
