import os
import sys
import csv
import json
import torch
import time

repo_root = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, repo_root)

from vllm import LLM, SamplingParams
from tqdm import tqdm
from tests import adapters  

model_dir = "/content/drive/MyDrive/Training files Assignment 3/Model Outputs/sft model"   
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
    files = [f for f in os.listdir(mmlu_dir) if f.endswith(".csv")]
    total_correct = 0
    total_examples = 0
    all_start = time.time()

    for filename in files:
        path_to_csv = os.path.join(mmlu_dir, filename)
        examples = load_mmlu_csv(path_to_csv)
        prompts = [format_instruction_prompt(ex) for ex in examples]
        labels = [ex["answer"] for ex in examples]

        model = LLM(model=model_dir, enforce_eager=True)
        sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens, stop=["\n"])

        print(f"\nEvaluating {len(prompts)} examples from {filename}")
        start_time = time.time()

        outputs = []
        for i in tqdm(range(0, len(prompts), 1)):  # 1 prompt at a time
            sub_outputs = model.generate(prompts[i:i+1], sampling_params=sampling_params)
            outputs.extend(sub_outputs)

        end_time = time.time()

        correct_count = 0
        results = []

        out_file = f"mmlu_outputs_{filename.replace('.csv', '')}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for ex, out, label, prompt in zip(examples, outputs, labels, prompts):
                raw_response = out.outputs[0].text.strip()
                prediction = next((ch for ch in raw_response if ch in "ABCD"), raw_response.split()[0] if raw_response else "")
                is_correct = prediction == label
                correct_count += int(is_correct)

                result = {
                    "question": ex["question"],
                    "options": ex["options"],
                    "correct": label,
                    "predicted": prediction,
                    "is_correct": is_correct,
                    "model_output": raw_response,
                    "prompt": prompt
                }
                results.append(result)
                f.write(json.dumps(result) + "\n")

        total_correct += correct_count
        total_examples += len(results)
        

    overall_time = time.time() - all_start
    overall_accuracy = total_correct / total_examples * 100
    overall_throughput = total_examples / overall_time
    print(f"Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_examples})")
    print(f"Overall Throughput: {overall_throughput:.2f} examples/sec")




if __name__ == "__main__":
    main()
