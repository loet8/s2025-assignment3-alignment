import os
os.environ['VLLM_USE_V1'] = '0'
import sys
import csv
import json

import multiprocessing as mp
mp.set_start_method("spawn", force=True)


import torch
import time

repo_root = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, repo_root)

from vllm import LLM, SamplingParams
from tqdm import tqdm
from tests import adapters  

model_dir = "/content/sft_model_local"   
mmlu_dir = "/content/s2025-assignment3-alignment/data/mmlu/dev"              
batch_size = 5
max_tokens = 512
out_file = "mmlu_sft_outputs.jsonl"


def load_mmlu_csv(csv_path: str):
    examples = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 6:
                continue
            example = {
                "question": row[0],
                "options": [row[1], row[2], row[3], row[4]],
                "answer": row[-1].strip().upper()
            }
            examples.append(example)
    return examples

def format_instruction_prompt(example):
    options = "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(example["options"])])
    full_question = f"{example['question']}\n\n{options}"
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{full_question}\n\n"
        "### Response:"
    )

def main():
    llm = LLM(model=model_dir, gpu_memory_utilization=0.7, max_num_seqs=2)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens, stop=["\n"])

    total_correct = 0
    total_examples = 0
    start_all = time.time()

    for fname in os.listdir(mmlu_dir):
        if not fname.endswith(".csv"):
            continue

        path = os.path.join(mmlu_dir, fname)
        examples = load_mmlu_csv(path)
        prompts = [format_instruction_prompt(ex) for ex in examples]
        labels = [ex["answer"] for ex in examples]

        outputs = []
        for i in tqdm(range(len(prompts))):
            out = llm.generate(prompts[i:i+1], sampling_params=sampling_params)
            outputs.extend(out)

        correct = 0
        out_path = f"mmlu_outputs_{fname.replace('.csv','.jsonl')}"
        with open(out_path, "w", encoding="utf-8") as fout:
            for ex, resp, lbl in zip(examples, outputs, labels):
                raw = resp.outputs[0].text.strip()
                pred = next((ch for ch in raw if ch in "ABCD"), raw.split()[0] if raw else "")
                is_corr = (pred == lbl)
                if is_corr:
                    correct += 1
                result = {
                    "question":     ex["question"],
                    "options":      ex["options"],
                    "correct":      lbl,
                    "predicted":    pred,
                    "is_correct":   is_corr,
                    "model_output": raw,
                }
                fout.write(json.dumps(result) + "\n")

        total_correct += correct
        total_examples += len(examples)

    overall_time = time.time() - start_all
    print(f"Overall Accuracy: {total_correct}/{total_examples} = {total_correct/total_examples*100:.2f}%")
    print(f"Overall Throughput: {total_examples/overall_time:.2f} ex/s")

if __name__ == "__main__":
    main()
