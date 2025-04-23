from __future__ import annotations
import argparse
import logging
import os
import time
import pathlib
from contextlib import nullcontext
import torch
import torch.nn.functional as F
import wandb
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from tests import adapters

logger = logging.getLogger(__name__)

def train(
    train_path,
    dev_path,
    output_dir,
    context_length,
    batch_size,
    train_steps,
    gradient_accumulation_steps,
    eval_iters,
    eval_interval,
    learning_rate,
    lr_scheduler,
    warmup_ratio,
    weight_decay,
    adam_beta1,
    adam_beta2,
    adam_eps,
    grad_clip,
    device,
    compile,
    dtype,
    wandb_project,
):
    tokenizer = AutoTokenizer.from_pretrained(train_path)
    model = AutoModelForCausalLM.from_pretrained(
        train_path,
        torch_dtype={
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[dtype],
    ).to(device)
    if compile:
        torch.set_float32_matmul_precision("high")
        model = torch.compile(model)
    is_ddp = int(os.environ.get("RANK", -1)) != -1
    if is_ddp:
        init_process_group(backend="nccl")
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        model = DDP(model, device_ids=[ddp_local_rank])
    dataset = adapters.get_packed_sft_dataset(tokenizer, train_path, context_length, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=not is_ddp, num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if p.dim() >= 2], "weight_decay": weight_decay},
            {"params": [p for p in model.parameters() if p.dim() < 2], "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
    )
    amp_ctx = nullcontext() if device.startswith("cpu") else torch.amp.autocast(device_type=device.split(":")[0], dtype=model.dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
    model.train()
    optimizer.zero_grad()
    global_step = 0
    start = time.time()
    it = iter(dataloader)
    for step in range(train_steps):
        for micro in range(gradient_accumulation_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dataloader)
                batch = next(it)
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            with amp_ctx:
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        if global_step % eval_interval == 0:
            dev_loss = 0.0
            model.eval()
            for _ in range(eval_iters):
                batch = next(it)
                with torch.no_grad():
                    outputs = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
                    dev_loss += outputs.loss.item()
            dev_loss /= eval_iters
            model.train()
            logger.info(f"Validation loss at step {global_step}: {dev_loss:.4f}")
        if global_step % gradient_accumulation_steps == 0 and global_step % gradient_accumulation_steps == 0:
            if global_step % gradient_accumulation_steps == 0:
                elapsed = time.time() - start
                logger.info(f"Step {global_step}/{train_steps} loss={loss.item()*gradient_accumulation_steps:.4f} time={elapsed:.1f}s")
        if global_step % gradient_accumulation_steps == 0 and global_step % gradient_accumulation_steps == 0:
            pass
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    if is_ddp:
        destroy_process_group()

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--dev-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--train-steps", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--eval-iters", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--lr-scheduler", type=str, choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.98)
    parser.add_argument("--adam-eps", type=float, default=1e-9)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--device", required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"], default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16")
    parser.add_argument("--wandb-project", type=str)
    args = parser.parse_args()
    is_ddp = int(os.environ.get("RANK", -1)) != -1
    is_master = int(os.environ.get("RANK", 0)) == 0 if is_ddp else True
    if is_master:
        if args.wandb_project:
            wandb.login()
            wandb.init(project=args.wandb_project, config=vars(args), name=pathlib.Path(args.output_dir).name)
    train(
        args.train_path,
        args.dev_path,
        args.output_dir,
        args.context_length,
        args.batch_size,
        args.train_steps,
        args.gradient_accumulation_steps,
        args.eval_iters,
        args.eval_interval,
        args.learning_rate,
        args.lr_scheduler,
        args.warmup_ratio,
        args.weight_decay,
        args.adam_beta1,
        args.adam_beta2,
        args.adam_eps,
        args.grad_clip,
        args.device,
        args.compile,
        args.dtype,
        args.wandb_project,
    )
    logger.info("finished running scripts/train.py")
