import os
import json
import time
from vllm import LLM, SamplingParams

os.environ["VLLM_DISABLE_USAGE_REPORTING"] = "1"

def load_alpaca_eval_examples(filepath: str):
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line.strip()))
    return examples

def main():
    filepath = "/Users/tiffanyloe/Desktop/ECE 491B/Assignment 3/s2025-assignment3-alignment/data/alpaca_eval/alpaca_eval.jsonl"
    examples = load_alpaca_eval_examples(filepath)
    instructions = [ex["instruction"] for ex in examples]
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024, stop=["\n"])
    model_path = "../Qwen/Qwen2.5-0.5B" 
    generator_name = "qwen2.5-0.5b"
    llm = LLM(model=model_path)
    batch_size = 5
    outputs = []
    start_time = time.time()
    for i in range(0, len(instructions), batch_size):
        batch_instructions = instructions[i:i+batch_size]
        batch_outputs = llm.generate(batch_instructions, sampling_params)
        outputs.extend(batch_outputs)
    elapsed_time = time.time() - start_time
    throughput = len(instructions) / elapsed_time if elapsed_time > 0 else 0.0
    predictions = []
    for ex, output in zip(examples, outputs):
        predictions.append({
            "instruction": ex["instruction"],
            "output": output.outputs[0].text.strip(),
            "generator": generator_name,
            "dataset": ex.get("dataset", "alpaca_eval")
        })
    with open("alpaca_eval_outputs.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    print("Throughput (examples/sec):", throughput)
    print("Results saved to alpaca_eval_outputs.json")

if __name__ == "__main__":
    main()
