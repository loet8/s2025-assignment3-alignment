import os
os.environ['VLLM_USE_V1'] = '0'

import multiprocessing as mp
mp.set_start_method("spawn", force=True)
import os
import json
import csv
import time
import sys

repo_root = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, repo_root)

from vllm import LLM, SamplingParams
from tqdm import tqdm
from tests import adapters  

model_dir = "/content/sft_model_local"
sst_path = "/content/s2025-assignment3-alignment/data/simple_safety_tests/simple_safety_tests.csv"
max_tokens = 1024
out_file = "sst_sft_outputs.jsonl"

def load_sst_examples(filepath):
    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            inst = row.get('prompts_final') or row.get('instruction')
            if inst and inst.strip():
                row['instruction'] = inst.strip()
                examples.append(row)
    return examples

def format_instruction_prompt(example):
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        "### Response:"
    )

def main():
    examples = load_sst_examples(sst_path)
    prompts = [format_instruction_prompt(ex) for ex in examples]

    model = LLM(model=model_dir, gpu_memory_utilization=0.7, max_num_seqs=2)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens, stop=["###"])

    start_time = time.time()
    outputs = []
    for i in tqdm(range(len(prompts))):
        sub_outputs = model.generate(prompts[i:i+1], sampling_params=sampling_params)
        outputs.extend(sub_outputs)
    end_time = time.time()

    with open(out_file, "w", encoding="utf-8") as f:
        for ex, out, prompt in zip(examples, outputs, prompts):
            response = out.outputs[0].text.strip()
            record = {
                "instruction": ex["instruction"],
                "response": response,
                "prompt": prompt
            }
            f.write(json.dumps(record) + "\n")

    throughput = len(prompts) / (end_time - start_time)
    print(f"Throughput: {throughput:.2f} examples/second")
    print(f"Results saved to: {out_file}")

if __name__ == "__main__":
    main()
