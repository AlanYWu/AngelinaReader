"""Fine-tune Qwen3-VL-8B on braille OCR with LoRA + SFTTrainer.

Usage:
    CUDA_VISIBLE_DEVICES=0 python vision_llm/train_qwen3vl.py \
        --model_path /data1/wy/models/Qwen3-VL-8B-Instruct \
        --train_data vision_llm/train.json \
        --val_data vision_llm/val.json \
        --output_dir vision_llm/qwen3vl_lora_output
"""

import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
)
from qwen_vl_utils import process_vision_info
from trl import SFTTrainer, SFTConfig


def load_json_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data1/wy/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="vision_llm/qwen3vl_lora_output")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_pixels", type=int, default=1003520,
                        help="Max pixels per image (default ~1003x1003)")
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28,
                        help="Min pixels per image")
    args = parser.parse_args()

    print("Loading model and processor...")
    processor = Qwen3VLProcessor.from_pretrained(
        args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    # LoRA config targeting language model layers
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
    )

    # Load data
    print("Loading training data...")
    train_raw = load_json_data(args.train_data)
    train_dataset = Dataset.from_list(train_raw)
    print(f"  Train: {len(train_dataset)} samples")

    eval_dataset = None
    if args.val_data and os.path.exists(args.val_data):
        val_raw = load_json_data(args.val_data)
        eval_dataset = Dataset.from_list(val_raw)
        print(f"  Val: {len(eval_dataset)} samples")

    # Training config
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset else "no",
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        report_to="none",
        dataset_text_field="",  # not used, we use dataset_kwargs
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # Collator that handles vision inputs
    def collate_fn(examples):
        texts = []
        image_inputs_list = []
        for ex in examples:
            messages = ex["messages"]
            # Apply chat template to get text with image placeholders
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

            # Use qwen_vl_utils to properly extract and process images
            images, videos = process_vision_info(messages)
            if images:
                image_inputs_list.extend(images)

        # Process with the VL processor
        batch = processor(
            text=texts,
            images=image_inputs_list if image_inputs_list else None,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )

        # Create labels: mask everything before the assistant response
        labels = batch["input_ids"].clone()
        # Mask padding
        labels[labels == processor.tokenizer.pad_token_id] = -100

        # Mask prompt tokens (everything before assistant content)
        for i in range(len(texts)):
            text = texts[i]
            assistant_marker = "<|im_start|>assistant\n"
            marker_pos = text.rfind(assistant_marker)
            if marker_pos >= 0:
                prompt_text = text[:marker_pos + len(assistant_marker)]
                prompt_ids = processor.tokenizer.encode(
                    prompt_text, add_special_tokens=False
                )
                labels[i, :len(prompt_ids)] = -100

        batch["labels"] = labels
        return batch

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        data_collator=collate_fn,
        processing_class=processor.tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model(os.path.join(args.output_dir, "final"))
    processor.save_pretrained(os.path.join(args.output_dir, "final"))
    print("Done!")


if __name__ == "__main__":
    main()
