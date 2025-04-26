import json
import random
from pathlib import Path
from typing import List, Dict

def _parse_convo(convo: str) -> List[tuple]:
    msgs = []
    for line in convo.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Human:"):
            msgs.append(("human", line[len("Human:"):].strip()))
        elif line.startswith("Assistant:"):
            msgs.append(("assistant", line[len("Assistant:"):].strip()))
    return msgs

def load_anthropic_hh_dataset(base_dir: str) -> List[Dict[str,str]]:
    base = Path(base_dir)
    examples: List[Dict[str,str]] = []

    for train_path in base.glob("*/train.jsonl"):
        split = train_path.parent.name
        raw_count = 0
        kept_count = 0

        print(f"→ Loading {train_path} (split={split})")
        with train_path.open("r", encoding="utf-8") as f:
            for line in f:
                raw_count += 1
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                chosen_msgs   = _parse_convo(data.get("chosen", ""))
                rejected_msgs = _parse_convo(data.get("rejected", ""))

                if len(chosen_msgs)==2 and len(rejected_msgs)==2:
                    (r0, q0), (r1, a1) = chosen_msgs
                    (s0, _),  (s1, b1) = rejected_msgs

                    if (r0, s0) == ("human","human") and (r1, s1) == ("assistant","assistant"):
                        if q0 == data["rejected"].splitlines()[1].split("Assistant:",1)[0].replace("Human:","").strip() \
                           or q0 == rejected_msgs[0][1]:
                            examples.append({
                                "instruction": q0,
                                "chosen":      a1,
                                "rejected":    b1,
                                "source":      split
                            })
                            kept_count += 1

        print(f"scanned {raw_count} lines, kept {kept_count} single-turn examples\n")

    print(f"Loaded a total of {len(examples)} examples across all splits.")
    return examples


if __name__ == "__main__":
    base_folder = "/Users/tiffanyloe/Desktop/ECE 491B/Assignment 3/s2025-assignment3-alignment/HH dataset"
    all_examples = load_anthropic_hh_dataset(base_folder)

    harmless = [e for e in all_examples if e["source"] == "harmless-base"]
    helpful = [e for e in all_examples if e["source"] != "harmless-base"]

    print("=== 3 Harmless Examples ===")
    for i, ex in enumerate(random.sample(harmless, 3), 1):
        print(f"\n[{i}] PROMPT:   {ex['instruction']!r}")
        print(f"     CHOSEN:  {ex['chosen']!r}")
        print(f"     REJECT: {ex['rejected']!r}")

    print("\n=== 3 Helpful Examples ===")
    for i, ex in enumerate(random.sample(helpful, 3), 1):
        print(f"\n[{i}] PROMPT:   {ex['instruction']!r}")
        print(f"     CHOSEN:  {ex['chosen']!r}")
        print(f"     REJECT: {ex['rejected']!r}")
