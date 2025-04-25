import os
os.environ['VLLM_USE_V1'] = '0'


import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import json
import time
import sys

repo_root = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, repo_root)

from vllm import LLM, SamplingParams
from tqdm import tqdm
from tests import adapters 


model_dir = "/content/sft_model_local"
alpaca_eval_path = "/content/s2025-assignment3-alignment/data/alpaca_eval/alpaca_eval.jsonl"
batch_size = 5
max_tokens = 1024
out_file = "alpaca_eval_sft_outputs.json"


def load_alpaca_eval_examples(filepath: str):
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                examples.append(data)
    return examples



def format_instruction_prompt(example):
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        "### Response:"
    )


def main():
    examples = load_alpaca_eval_examples(alpaca_eval_path)
    prompts = [format_instruction_prompt(ex) for ex in examples]

    model = LLM(model=model_dir, gpu_memory_utilization=0.7, max_num_seqs=2)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens, stop=["###"])

    print(f"Evaluating {len(prompts)} AlpacaEval examples...")
    start_time = time.time()
    outputs = []
    for i in tqdm(range(0, len(prompts), 1)): 
        sub_outputs = model.generate(prompts[i:i+1], sampling_params=sampling_params)
        outputs.extend(sub_outputs)
    end_time = time.time()
    
    records = []
    
    for ex, out, prompt in zip(examples, outputs, prompts):
        response = out.outputs[0].text.strip()
        records.append({
            "instruction": ex["instruction"],
            "input": ex.get("input", ""),  
            "output": response, 
            "generator": "my_sft_model", 
            "prompt": prompt
        })

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    throughput = len(prompts) / (end_time - start_time)
    print(f"Throughput: {throughput:.2f} examples/second")
    print(f"Results saved to: {out_file}")

if __name__ == "__main__":
    main()
