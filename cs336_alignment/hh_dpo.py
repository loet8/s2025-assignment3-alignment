import json
from pathlib import Path
from typing import List, Dict

def load_anthropic_hh_dataset(base_dir: str) -> List[Dict]:
    base_path = Path(base_dir)
    subfolders = [
        "harmless-base",
        "helpful-base",
        "helpful-online",
        "helpful-rejection-sampled"
    ]
    
    examples = []

    for folder in subfolders:
        path = base_path / folder / "train.jsonl"
        if not path.exists():
            continue
        
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    chosen = data.get("chosen", [])
                    rejected = data.get("rejected", [])

                    if len(chosen) != 2 or len(rejected) != 2:
                        continue
                    if (
                        chosen[0]["role"] != "human" or chosen[1]["role"] != "assistant" or
                        rejected[0]["role"] != "human" or rejected[1]["role"] != "assistant"
                    ):
                        continue
                    if chosen[0]["content"] != rejected[0]["content"]:
                        continue

                    examples.append({
                        "instruction": chosen[0]["content"],
                        "chosen": chosen[1]["content"],
                        "rejected": rejected[1]["content"],
                        "source": folder
                    })
                except:
                    continue

    return examples
