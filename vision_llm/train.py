#!/usr/bin/env python
"""Training script for Braille VLM (Phase 2).

Trains the VisualTokenAdapter (17.8M params) and optionally LoRA on Qwen3-8B
to transcribe braille images into unicode braille text.

Usage:
    PYTHONPATH=. python vision_llm/train.py \
        --checkpoint weights/model.t7 \
        --params weights/param.txt \
        --llm Qwen/Qwen3-8B \
        --device cuda \
        --epochs 20 \
        --batch_size 2 \
        --lr 1e-4 \
        --use_lora
"""

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from vision_llm.braille_vlm import BrailleVLM, IMAGE_PLACEHOLDER
from vision_llm.config import VisualFeatureConfig
from vision_llm.data import BrailleVLMDataset


def collate_fn(batch, tokenizer, max_length=512):
    """Collate batch: tokenize prompts + targets, build labels."""
    images = torch.stack([item["image"] for item in batch])

    # Build full text: <image>\n{prompt}\n{target}<eos>
    full_texts = []
    prompt_texts = []
    for item in batch:
        prompt = f"{IMAGE_PLACEHOLDER}\n{item['prompt']}\n"
        full = prompt + item["target"] + tokenizer.eos_token
        full_texts.append(full)
        prompt_texts.append(prompt)

    # Tokenize full sequence
    tokenized = tokenizer(
        full_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # Build labels: mask prompt tokens with -100, only compute loss on target
    labels = input_ids.clone()
    for i, prompt_text in enumerate(prompt_texts):
        prompt_tokens = tokenizer(
            prompt_text, add_special_tokens=False, return_tensors="pt"
        )
        prompt_len = prompt_tokens["input_ids"].shape[1]
        labels[i, :prompt_len] = -100
    # Mask padding
    labels[attention_mask == 0] = -100

    return {
        "images": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def setup_lora(model, lora_r=16, lora_alpha=32, lora_dropout=0.05):
    """Apply LoRA to Qwen3's attention layers."""
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.llm = get_peft_model(model.llm, lora_config)
    # Unfreeze adapter (LoRA setup may have affected requires_grad)
    for p in model.adapter.parameters():
        p.requires_grad = True
    return model


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, grad_accum_steps=1):
    model.train()
    # Feature extractor stays frozen/eval
    model.feature_extractor.eval()

    total_loss = 0.0
    num_steps = 0
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            num_steps += 1

        total_loss += outputs.loss.item()

        if (step + 1) % 10 == 0:
            avg_loss = total_loss / (step + 1)
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch} | Step {step+1}/{len(dataloader)} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    for batch in dataloader:
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss += outputs.loss.item()

    return total_loss / max(len(dataloader), 1)


def main():
    parser = argparse.ArgumentParser(description="Train Braille VLM")
    parser.add_argument("--checkpoint", default="weights/model.t7", help="RetinaNet weights")
    parser.add_argument("--params", default="weights/param.txt", help="RetinaNet params")
    parser.add_argument("--llm", default="Qwen/Qwen3-8B", help="LLM model name or path")
    parser.add_argument("--device", default="cuda", help="Device: cpu, cuda, mps")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--use_lora", action="store_true", help="Apply LoRA to Qwen3")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", default="vision_llm_checkpoints")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Config ---
    config = VisualFeatureConfig(
        checkpoint_path=args.checkpoint,
        param_path=args.params,
        device=args.device,
    )

    print(f"Visual tokens: {config.num_visual_tokens}, LLM dim: {config.llm_hidden_dim}")

    # --- Model ---
    print(f"Loading Braille VLM with {args.llm}...")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = BrailleVLM(
        config=config,
        llm_name_or_path=args.llm,
        freeze_llm=True,  # Frozen by default; LoRA unfreezes specific params
        torch_dtype=dtype,
    )

    if args.use_lora:
        print("Applying LoRA to Qwen3...")
        model = setup_lora(model, lora_r=args.lora_r, lora_alpha=args.lora_alpha)

    model = model.to(device)
    trainable = model.get_trainable_params()
    print(f"Trainable params — adapter: {trainable['adapter']:,}, llm: {trainable['llm']:,}, total: {trainable['total']:,}")

    # --- Data ---
    data_root = str(Path(__file__).resolve().parent.parent)
    print(f"Loading data from {data_root}...")

    train_dataset = BrailleVLMDataset(data_root, image_size=args.image_size, split="train")
    val_dataset = BrailleVLMDataset(data_root, image_size=args.image_size, split="val")
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    tokenizer = model.tokenizer
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, args.max_length),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, args.max_length),
    )

    # --- Optimizer ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # --- Training loop ---
    best_val_loss = float("inf")
    print(f"\nStarting training: {args.epochs} epochs, {total_steps} optimizer steps")
    print(f"Batch size: {args.batch_size}, Grad accum: {args.grad_accum}, Effective batch: {args.batch_size * args.grad_accum}")

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, args.grad_accum
        )
        val_loss = validate(model, val_loader, device)
        print(f"Epoch {epoch} — Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = output_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "adapter_state_dict": model.adapter.state_dict(),
                "val_loss": val_loss,
                "config": vars(config),
            }, save_path)
            # Save LoRA weights separately if applicable
            if args.use_lora:
                model.llm.save_pretrained(str(output_dir / "best_lora"))
            print(f"  Saved best model (val_loss={val_loss:.4f})")

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            save_path = output_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "adapter_state_dict": model.adapter.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, save_path)
            if args.use_lora:
                model.llm.save_pretrained(str(output_dir / f"lora_epoch{epoch}"))

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in {output_dir}")


if __name__ == "__main__":
    main()
