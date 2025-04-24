from __future__ import annotations
import argparse
import json
import logging
import os
import time
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
import wandb
from tests import adapters

logger = logging.getLogger(__name__)


def train(
    train_path,
    dev_path,
    output_dir,
    model_name,
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
    is_ddp = int(os.environ.get("RANK", -1)) != -1
    if is_ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        seed = ddp_rank
        is_master = ddp_rank == 0
    else:
        ddp_world_size = 1
        seed = 0
        is_master = True
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype={"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype],
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device)
    if compile:
        torch.set_float32_matmul_precision("high")
        model = torch.compile(model)
    if is_ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    if is_master:
        os.makedirs(output_dir, exist_ok=True)
        cfg_path = os.path.join(output_dir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(model.config.to_diff_dict(), f, indent=2)
        if wandb_project:
            wandb.login()
            wandb.init(project=wandb_project, config=vars(), name=os.path.basename(output_dir))
    train_ds = adapters.get_packed_sft_dataset(tokenizer, train_path, context_length, shuffle=True)
    dev_ds = adapters.get_packed_sft_dataset(tokenizer, dev_path, context_length, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=not is_ddp, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)
    param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
    params_decay = [p for p in param_dict.values() if p.dim() >= 2]
    params_no_decay = [p for p in param_dict.values() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": params_decay, "weight_decay": weight_decay},
         {"params": params_no_decay, "weight_decay": 0.0}],
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
    )
    if lr_scheduler.lower() == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(train_steps * warmup_ratio),
            num_training_steps=train_steps,
        )
    else:
        scheduler = None
    amp_ctx = nullcontext() if device.startswith("cpu") else torch.amp.autocast(device_type=device.split(":")[0], dtype=model.dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
    model.train()
    optimizer.zero_grad()
    train_iter = iter(train_loader)
    start_time = time.time()
    for step in tqdm(range(train_steps), desc="Training"):
        for micro in range(gradient_accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            if is_ddp:
                model.require_backward_grad_sync = (micro == gradient_accumulation_steps - 1)
            with amp_ctx:
                outputs = model(input_ids=inputs, labels=labels)
                loss = outputs.loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
        else:
            lr = learning_rate
        if is_master and ((step + 1) % gradient_accumulation_steps == 0):
            train_loss = loss.item() * gradient_accumulation_steps
            elapsed = time.time() - start_time
            logger.info(f"Step {step+1}/{train_steps}  loss={train_loss:.4f}  lr={lr:.3e}  time={elapsed:.1f}s")
            if wandb_project:
                wandb.log({"train_loss": train_loss, "lr": lr}, step=step+1)
        if is_master and ((step + 1) % eval_interval == 0 or step + 1 == train_steps):
            val_loss = estimate_dev_loss(model, dev_loader, eval_iters, device)
            logger.info(f"→ Eval @ step {step+1}: val_loss={val_loss:.4f}")
            if wandb_project:
                wandb.log({"eval_loss": val_loss}, step=step+1)
    if is_master:
        save_model = model.module if isinstance(model, DDP) else model
        save_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    if is_ddp:
        destroy_process_group()


@torch.no_grad()
def estimate_dev_loss(model, dev_loader, eval_iters, device):
    model.eval()
    total_loss = 0.0
    it = iter(dev_loader)
    for k in range(eval_iters):
        try:
            batch = next(it)
        except StopIteration:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        total_loss += outputs.loss.item()
    model.train()
    return total_loss / max(1, k + 1)

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-path", required=True, help="Path to instruction-tune JSONL (train)")
    parser.add_argument("--dev-path", required=True, help="Path to instruction-tune JSONL (dev)")
    parser.add_argument("--output-dir", required=True, help="Where to save model+tokenizer")
    parser.add_argument("--model-name", required=True, help="HF model ID or path (e.g. Qwen/Qwen2.5-0.5B)")
    parser.add_argument("--context-length", type=int, required=True, help="Max seq length")
    parser.add_argument("--batch-size", type=int, required=True, help="Per-device batch size")
    parser.add_argument("--train-steps", type=int, required=True, help="Total optimizer updates")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Accumulate grads over N micro-batches")
    parser.add_argument("--eval-iters", type=int, default=200, help="Dev batches per eval")
    parser.add_argument("--eval-interval", type=int, default=2000, help="Evaluate every N updates")
    parser.add_argument("--learning-rate", type=float, required=True, help="Initial AdamW LR")
    parser.add_argument("--lr-scheduler", type=str, choices=["constant","cosine"], default="cosine", help="LR schedule")
    parser.add_argument("--warmup-ratio", type=float, default=0.01, help="Linear warmup fraction")
    parser.add_argument("--weight-decay", type=float, default=1e-1, help="AdamW weight decay")
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam-beta2", type=float, default=0.98, help="Adam beta2")
    parser.add_argument("--adam-eps", type=float, default=1e-9, help="Adam epsilon")
    parser.add_argument("--grad-clip", type=float, default=None, help="Clip grads to this norm")
    parser.add_argument("--device", required=True, help="e.g. 'cpu', 'cuda', 'cuda:0'")
    parser.add_argument("--compile", action="store_true", help="Torch 2.0 compile(model)")
    parser.add_argument("--dtype", choices=["float32","float16","bfloat16"], default="bfloat16", help="Model dtype")
    parser.add_argument("--wandb-project", default=None, help="Log to this W&B project (optional)")
    args = parser.parse_args()
    train(
        args.train_path,
        args.dev_path,
        args.output_dir,
        args.model_name,
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
    logger.info("Finished training.")
