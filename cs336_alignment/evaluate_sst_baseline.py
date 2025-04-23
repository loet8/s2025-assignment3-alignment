
import os
import json
import csv
import logging
from vllm import LLM, SamplingParams
from tqdm import tqdm

os.environ["VLLM_DISABLE_USAGE_REPORTING"] = "1"

def load_sst_examples(filepath):
    """
    Load SimpleSafetyTests examples from a CSV file.
    Expects a 'prompts_final' column.
    Returns a list of dicts (one per row).
    """
    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            inst = row.get('prompts_final') or row.get('instruction')
            if inst and inst.strip():
                examples.append(row)
    return examples

def main():
    input_csv   = "/content/s2025-assignment3-alignment/data/simple_safety_tests/simple_safety_tests.csv"
    model_path  = "/content/Qwen/Qwen2.5-0.5B"
    output_file = "qwen_sst_baseline_outputs.jsonl"
    batch_size  = 8

    logging.basicConfig(level=logging.INFO)

    examples     = load_sst_examples(input_csv)
    instructions = [row["prompts_final"].strip() for row in examples]

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["\n"]
    )
    llm = LLM(model=model_path)

    with open(output_file, 'w', encoding='utf-8') as fout:
        for i in tqdm(range(0, len(instructions), batch_size), desc="Generating"):
            batch = instructions[i : i + batch_size]
            results = llm.generate(batch, sampling_params)
            for row, result in zip(examples[i : i + batch_size], results):
                generated = result.outputs[0].text.strip()
                fout.write(json.dumps({
                    "prompts_final": row["prompts_final"].strip(),
                    "output":        generated
                }) + "\n")

    logging.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
