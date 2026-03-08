"""Inference with fine-tuned Qwen3-VL for braille OCR.

Usage:
    python vision_llm/infer_qwen3vl.py \
        --model_path /data1/wy/models/Qwen3-VL-8B-Instruct \
        --lora_path vision_llm/qwen3vl_lora_output/final \
        --val_data vision_llm/val.json \
        --output_file vision_llm/qwen3vl_eval_results.md
"""

import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data1/wy/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="vision_llm/qwen3vl_eval_results.md")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    print("Loading model...")
    processor = Qwen3VLProcessor.from_pretrained(args.model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()

    print("Loading val data...")
    with open(args.val_data, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    results = []
    n = min(args.num_samples, len(val_data))

    for i in range(n):
        sample = val_data[i]
        messages = sample["messages"]
        ground_truth = messages[1]["content"][0]["text"]

        # Build inference messages (user turn only)
        infer_messages = [messages[0]]
        text = processor.apply_chat_template(
            infer_messages, tokenize=False, add_generation_prompt=True
        )

        # Extract image
        image_url = messages[0]["content"][0]["image"]
        inputs = processor(
            text=[text],
            images=[image_url],
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Decode only generated tokens
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        # Strip thinking tags if present
        if "<think>" in output_text:
            think_end = output_text.rfind("</think>")
            if think_end >= 0:
                output_text = output_text[think_end + len("</think>"):].strip()

        image_name = os.path.basename(image_url.replace("file://", ""))
        print(f"\n--- Sample {i+1}: {image_name} ---")
        print(f"GT:  {ground_truth[:80]}...")
        print(f"Out: {output_text[:80]}...")

        # Character-level accuracy
        gt_chars = list(ground_truth.replace("\n", "").replace(" ", ""))
        out_chars = list(output_text.replace("\n", "").replace(" ", ""))
        if gt_chars:
            matches = sum(1 for a, b in zip(gt_chars, out_chars) if a == b)
            char_acc = matches / len(gt_chars) * 100
        else:
            char_acc = 0.0

        results.append({
            "image": image_name,
            "ground_truth": ground_truth,
            "output": output_text,
            "char_accuracy": char_acc,
        })

    # Write results
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("# Qwen3-VL Braille OCR Evaluation\n\n")
        f.write(f"| Sample | Char Accuracy | Issue |\n")
        f.write(f"|--------|--------------|-------|\n")
        for r in results:
            issue = "OK" if r["char_accuracy"] > 50 else "Low accuracy"
            if len(set(r["output"])) < 5:
                issue = "Repetitive"
            f.write(f"| {r['image'][:40]} | {r['char_accuracy']:.1f}% | {issue} |\n")

        f.write("\n## Detailed Results\n\n")
        for i, r in enumerate(results):
            f.write(f"### Sample {i+1}: {r['image']}\n\n")
            f.write(f"**Ground Truth:**\n```\n{r['ground_truth']}\n```\n\n")
            f.write(f"**Model Output:**\n```\n{r['output']}\n```\n\n")
            f.write(f"**Char Accuracy:** {r['char_accuracy']:.1f}%\n\n")

    print(f"\nResults written to {args.output_file}")
    avg_acc = sum(r["char_accuracy"] for r in results) / len(results) if results else 0
    print(f"Average char accuracy: {avg_acc:.1f}%")


if __name__ == "__main__":
    main()
