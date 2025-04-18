import os
import json
import time
import logging
from vllm import LLM, SamplingParams

os.environ["VLLM_DISABLE_USAGE_REPORTING"] = "1"

def load_alpaca_eval_examples(filepath: str):
    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples

def main():
    filepath = "/Users/tiffanyloe/Desktop/ECE 491B/Assignment 3/s2025-assignment3-alignment/data/alpaca_eval/alpaca_eval.jsonl"
    eval_set = load_alpaca_eval_examples(filepath)

    model_path = "../Qwen/Qwen2.5-0.5B" 
    generator_name = "qwen2.5-0.5b"

    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=2048, stop=["\n"])
    
    llm = LLM(model=model_path)

    batch_size = 5
    all_outputs = []
    start_time = time.time()
    
    for i in range(0, len(eval_set), batch_size):
        batch = eval_set[i:i + batch_size]
        instructions = [ex['instruction'] for ex in batch]
        # Generate completions
        results = llm.generate(instructions, sampling_params)
        for ex, out in zip(batch, results):
            text = out.outputs[0].text.strip()
            all_outputs.append({
                "instruction": ex['instruction'],
                "output": text,
                "generator": generator_name,
                "dataset": ex.get('dataset', 'alpaca_eval')
            })
    elapsed_time = time.time() - start_time
    logging.info(f"Generated {len(all_outputs)} outputs in {elapsed_time:.1f}s ({len(all_outputs)/elapsed_time:.1f} ex/s)")

    out_file = "alpaca_zero_shot_outputs.json"
    with open(out_file, 'w', encoding='utf-8') as fout:
        json.dump(all_outputs, fout, ensure_ascii=False, indent=2)

    logging.info(f"Wrote zero-shot predictions to {out_file}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
