import os
import csv
import json
import torch
from typing import Any, List, Dict
from vllm import LLM, SamplingParams
from tests import adapters

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

def main():
    mmlu_base_dir = "/Users/tiffanyloe/Desktop/ECE 491B/Assignment 3/s2025-assignment3-alignment/data/mmlu/dev"
    examples = load_all_examples(mmlu_base_dir)
    print(f"Loaded {len(examples)} MMLU examples.")
    prompts = [format_prompt(example) for example in examples]
    model_path = "../Qwen/Qwen2.5-0.5B"
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,
        stop=["\n"]
    )
    llm = LLM(model=model_path)
    batch_size = 5
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        print(f"Processing batch {i // batch_size + 1} / {((len(prompts)-1) // batch_size) + 1}")
        batch_outputs = llm.generate(batch_prompts, sampling_params)
        outputs.extend(batch_outputs)
        torch.cuda.empty_cache()
    num_correct = 0
    results = []
    for example, prompt, output in zip(examples, prompts, outputs):
        model_output = output.outputs[0].text
        predicted = adapters.run_parse_mmlu_response(example, model_output)
        correct_answer = example["answer"]
        is_correct = (predicted == correct_answer)
        if is_correct:
            num_correct += 1
        results.append({
            "example": example,
            "prompt": prompt,
            "model_output": model_output,
            "predicted": predicted,
            "correct_answer": correct_answer,
            "is_correct": is_correct
        })
    accuracy = num_correct / len(examples) if examples else 0.0
    print(f"Accuracy: {accuracy * 100:.2f}% ({num_correct}/{len(examples)})")
    results_outfile = "mmlu_results.json"
    with open(results_outfile, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": accuracy,
            "num_correct": num_correct,
            "total_examples": len(examples),
            "results": results
        }, f, indent=2)
    print(f"Results saved to {results_outfile}")

if __name__ == "__main__":
    main()