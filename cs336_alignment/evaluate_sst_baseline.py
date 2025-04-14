import os
import json
import csv
import time
import logging
from vllm import LLM, SamplingParams


os.environ["VLLM_DISABLE_USAGE_REPORTING"] = "1"

def load_sst_examples(filepath: str):
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append(row)
    return examples

def main():
    filepath = "/Users/tiffanyloe/Desktop/ECE 491B/Assignment 3/s2025-assignment3-alignment/data/simple_safety_tests/simple_safety_tests.csv"
    examples = load_sst_examples(filepath)
    instructions = [ex["prompts_final"] for ex in examples]
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024, stop=["\n"])
    model_path = "../Qwen/Qwen2.5-0.5B"
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
        prediction = {
            "prompts_final": ex["prompts_final"],
            "output": output.outputs[0].text.strip()
        }
        predictions.append(prediction)

    output_filename = "simplesafety_outputs.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    logging.info("Throughput (examples/sec):", throughput)
    logging.info("Results saved to", output_filename)

if __name__ == "__main__":
    main()