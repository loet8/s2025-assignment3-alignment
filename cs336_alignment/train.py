import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AdamW,
    get_scheduler
)
from tqdm import tqdm
from tests import adapters

try:
    import wandb
    USE_WANDB = True
except ImportError:
    USE_WANDB = False


config = {
    "model_name_or_path": "/path/to/qwen2.5-0.5b",
    "tokenizer_name_or_path": "/path/to/qwen2.5-0.5b",
    "train_file": "/path/to/train.jsonl",
    "output_dir": "./qwen_sft_output",

    "seq_length": 512,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "max_steps": 200,
    "logging_steps": 10,

    "use_flashattention": False, 
    "use_bfloat16": False,        
    "batch_size": 1,              
    "gradient_accumulation_steps": 32,
    "wandb_project": "sft_qwen",  
}


def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config["wandb_project"] and USE_WANDB:
        wandb.init(project=config["wandb_project"], config=config)

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name_or_path"])

    model_args = {
        "torch_dtype": torch.bfloat16 if config["use_bfloat16"] else torch.float32
    }
    if config["use_flashattention"]:
        model_args["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        **model_args
    ).to(device)

    dataset = adapters.get_packed_sft_dataset(
        tokenizer,
        config["train_file"],
        seq_length=config["seq_length"],
        shuffle=True
    )
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=int(config["max_steps"] * config["warmup_ratio"]),
        num_training_steps=config["max_steps"]
    )

    model.train()
    step = 0
    model.zero_grad()

    print(f"Starting training on {device} for {config['max_steps']} steps...")

    while step < config["max_steps"]:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids).logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=tokenizer.pad_token_id
            ) / config["gradient_accumulation_steps"]

            loss.backward()

            if (step + 1) % config["gradient_accumulation_steps"] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % config["logging_steps"] == 0:
                print(f"Step {step}: loss = {loss.item():.4f}")
                if config["wandb_project"] and USE_WANDB:
                    wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]}, step=step)

            step += 1
            if step >= config["max_steps"]:
                break

    os.makedirs(config["output_dir"], exist_ok=True)
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    print(f"Training complete. Model saved to {config['output_dir']}")


if __name__ == "__main__":
    train(config)
