"""Inference with fine-tuned Qwen3-VL for braille OCR.

Supports three modes:
  --mode vision_only   : image + "Transcribe the braille" (original)
  --mode hybrid        : image + RetinaNet detection hint + "Verify and correct"
  --mode retinanet_only: RetinaNet detection text only (no image)

In hybrid/retinanet_only modes, if --val_data already contains detection
hints in the prompt (from prepare_qwen3vl_data.py --mode hybrid), those
are used directly. Otherwise, RetinaNet is run live on each image.

Usage:
    PYTHONPATH=. python vision_llm/infer_qwen3vl.py \
        --model_path /data1/wy/models/Qwen3-VL-8B-Instruct \
        --lora_path vision_llm/qwen3vl_lora_output/final \
        --val_data vision_llm/val_hybrid.json \
        --mode hybrid \
        --output_file vision_llm/qwen3vl_hybrid_eval_results.md
"""

import argparse
import json
import os

import torch
from PIL import Image
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

# Same prompt templates as prepare_qwen3vl_data.py
PROMPT_HYBRID = (
    "Transcribe the braille in this image. "
    "A braille detection model produced the following preliminary transcription:\n"
    "```\n{detection}\n```\n"
    "Verify and correct this transcription based on the image. "
    "Output only the corrected braille unicode text."
)

PROMPT_RETINANET_ONLY = (
    "The following braille unicode text was produced by an OCR detection model "
    "and may contain errors. Correct any errors and output the cleaned braille text:\n"
    "```\n{detection}\n```"
)


def run_retinanet_on_image(inferencer, image_path, lang):
    """Run RetinaNet inference on a single image and return braille unicode text."""
    results = inferencer.run(
        image_path, lang=lang, draw_refined=inferencer.DRAW_REFINED,
        find_orientation=True, process_2_sides=False,
        align_results=True,
    )
    if results is None:
        return ""
    out_braille = results.get('braille', [])
    if out_braille:
        return "\n".join(out_braille)
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data1/wy/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="vision_llm/qwen3vl_eval_results.md")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--mode", type=str, default="hybrid",
                        choices=["vision_only", "hybrid", "retinanet_only"],
                        help="Inference mode (should match training mode)")
    parser.add_argument("--retinanet_device", type=str, default="cuda:0",
                        help="Device for RetinaNet (only used if val_data lacks detection hints)")
    parser.add_argument("--lang", type=str, default="EN")
    parser.add_argument("--max_pixels", type=int, default=2016 * 2016,
                        help="Max pixels per image")
    parser.add_argument("--min_pixels", type=int, default=512 * 28 * 28,
                        help="Min pixels per image")
    args = parser.parse_args()

    print("Loading model...")
    processor = Qwen3VLProcessor.from_pretrained(
        args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    if args.lora_path:
        model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()

    print("Loading val data...")
    with open(args.val_data, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    # Check if val_data already has detection hints (from hybrid data prep)
    val_has_hints = False
    if val_data and args.mode in ("hybrid", "retinanet_only"):
        first_prompt = val_data[0]["messages"][0]["content"][-1].get("text", "")
        val_has_hints = "preliminary transcription" in first_prompt or "OCR detection model" in first_prompt

    # Load RetinaNet if needed and val_data doesn't already have hints
    inferencer = None
    if args.mode in ("hybrid", "retinanet_only") and not val_has_hints:
        print(f"Loading RetinaNet on {args.retinanet_device} for live detection...")
        from model.infer_retinanet import BrailleInference
        inferencer = BrailleInference(device=args.retinanet_device, verbose=0)
        print("RetinaNet loaded.")

    results = []
    n = min(args.num_samples, len(val_data))

    for i in range(n):
        sample = val_data[i]
        messages = sample["messages"]
        ground_truth = messages[1]["content"][0]["text"]

        # Find image path from user content
        image_path = None
        for item in messages[0]["content"]:
            if item.get("type") == "image":
                image_path = item["image"].replace("file://", "")
                break

        if val_has_hints:
            # Val data already has the right prompt format — use as-is
            infer_messages = [messages[0]]
        else:
            # Build prompt from scratch based on mode
            user_content = []
            if args.mode == "hybrid" and image_path:
                detection = run_retinanet_on_image(inferencer, image_path, args.lang)
                user_content.append({"type": "image", "image": f"file://{os.path.abspath(image_path)}"})
                user_content.append({"type": "text", "text": PROMPT_HYBRID.format(detection=detection)})
            elif args.mode == "retinanet_only" and image_path:
                detection = run_retinanet_on_image(inferencer, image_path, args.lang)
                user_content.append({"type": "text", "text": PROMPT_RETINANET_ONLY.format(detection=detection)})
            else:
                # vision_only — use original message as-is
                infer_messages = [messages[0]]
                user_content = None

            if user_content is not None:
                infer_messages = [{"role": "user", "content": user_content}]

        text = processor.apply_chat_template(
            infer_messages, tokenize=False, add_generation_prompt=True
        )

        # Process image if present
        img = None
        if image_path and args.mode != "retinanet_only":
            img = Image.open(image_path).convert("RGB")

        inputs = processor(
            text=[text],
            images=[img] if img else None,
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

        image_name = os.path.basename(image_path) if image_path else f"sample_{i}"
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
        f.write(f"# Qwen3-VL Braille OCR Evaluation ({args.mode} mode)\n\n")
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
