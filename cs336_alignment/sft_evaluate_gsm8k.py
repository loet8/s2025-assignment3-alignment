import os
os.environ['VLLM_USE_V1'] = '0'

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import sys
import json
import time

repo_root = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, repo_root)

from vllm import LLM, SamplingParams
from tqdm import tqdm
from typing import Any, List, Dict
from tests import adapters  

model_dir = "/content/sft_model_local"
gsm8k_path = "/content/s2025-assignment3-alignment/data/gsm8k/test.jsonl"
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

    model = LLM(model=model_dir, gpu_memory_utilization=0.7, max_num_seqs=2)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens, stop=["\n"])

    outputs = []
    for i in tqdm(range(len(prompts))):
        sub_outputs = model.generate(prompts[i:i+1], sampling_params=sampling_params)
        outputs.extend(sub_outputs)

    correct_count = 0
    results = []

    with open(out_file, "w", encoding="utf-8") as f:
        for ex, out, label in zip(examples, outputs, labels):
            raw_response = out.outputs[0].text.strip()
            parts = raw_response.split()
            predicted = parts[0].strip().strip(".") if parts else ""
            is_correct = label.startswith(predicted)
            correct_count += int(is_correct)
            result = {
                "question": ex["question"],
                "correct":  label,
                "predicted": predicted,
                "is_correct": is_correct,
                "model_output": raw_response
            }
            f.write(json.dumps(result) + "\n")

    total = len(examples)
    accuracy = correct_count / total * 100
    throughput = total / (time.time() - start_time)

    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{total})")
    print(f"Throughput: {throughput:.2f} examples/sec")
    print(f"Results saved to: {out_file}")

if __name__ == "__main__":
    start_time = time.time()
    main()
