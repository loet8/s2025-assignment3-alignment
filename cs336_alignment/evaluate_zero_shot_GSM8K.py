import os
import re
import json
from typing import List, Dict, Optional
from tests import adapters
from vllm import LLM, SamplingParams

os.environ["VLLM_DISABLE_USAGE_REPORTING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



def extract_gold_numeric_answer(gold_answer: str) -> Optional[str]:
    numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", gold_answer)
    return numbers[-1] if numbers else None

def load_gsm8k_examples(filepath: str) -> List[Dict]:
    examples = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line.strip()))
    except Exception as e:
        print(f"Error loading GSM8K examples from {filepath}: {e}")
    return examples

def evaluate_predictions(examples: List[Dict], predictions: List[Dict]) -> Dict:
    total = len(predictions)
    correct = 0
    for pred in predictions:
        gold_raw = pred.get("gold_answer")
        gold = extract_gold_numeric_answer(gold_raw)
        parsed = pred.get("parsed_answer")
        try:
            if parsed is not None and gold is not None and abs(float(parsed) - float(gold)) < 1e-9:
                correct += 1
        except Exception:
            continue
    accuracy = correct / total if total > 0 else 0
    return {"accuracy": accuracy, "total": total, "correct": correct}

def main():
    filepath = "/Users/tiffanyloe/Desktop/ECE 491B/Assignment 3/s2025-assignment3-alignment/data/gsm8k/train.jsonl"
    examples = load_gsm8k_examples(filepath)
    if not examples:
        print("No examples loaded. Please check the train.jsonl file path and format.")
        return
    prompts = [f"{ex['question']}\n\nAnswer:" for ex in examples]
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024, stop=["\n"])
    model_path = "../Qwen/Qwen2.5-0.5B"
    llm = LLM(model=model_path)
    batch_size = 5
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_outputs = llm.generate(batch_prompts, sampling_params)
        outputs.extend(batch_outputs)
    evaluation_results = []
    for ex, prompt, output in zip(examples, prompts, outputs):
        model_output = output.outputs[0].text
        parsed_answer = adapters.run_parse_gsm8k_response(model_output)
        evaluation_results.append({
            "question": ex["question"],
            "gold_answer": ex["answer"].strip(),
            "prompt": prompt,
            "model_output": model_output,
            "parsed_answer": parsed_answer,
        })
    metrics = evaluate_predictions(examples, evaluation_results)
    print("Evaluation Metrics:")
    print(metrics)
    output_filename = "gsm8k_evaluation_results.json"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump({"evaluation_results": evaluation_results, "metrics": metrics}, f, indent=2)
        print(f"Results saved to {output_filename}")
    except Exception as e:
        print(f"Error saving results to {output_filename}: {e}")

if __name__ == "__main__":
    main()
