"""Prepare training data for Qwen3-VL fine-tuning.

Converts AngelinaDataset + DSBI annotations into Qwen3-VL chat format JSON.
Run from the AngelinaReader root directory with PYTHONPATH set.

Modes:
  --mode vision_only   : image + "Transcribe the braille" (original)
  --mode hybrid        : image + RetinaNet detection hint + "Verify and correct"
  --mode retinanet_only: RetinaNet detection text only (no image), LLM error correction

In hybrid/retinanet_only modes, RetinaNet is run on each image to produce
a preliminary transcription that is included in the prompt. The model learns
to verify/correct these detections using visual context (hybrid) or pure
language understanding (retinanet_only).

Usage:
    PYTHONPATH=. python vision_llm/prepare_qwen3vl_data.py \
        --data_root . --mode hybrid --device mps
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision_llm.data import collect_samples, _read_labelme_annotations, _read_dsbi_annotations, rects_to_braille_text
import PIL.Image

# Prompt templates
PROMPT_VISION_ONLY = "Transcribe the braille in this image."

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
    # draw_results() stores braille unicode lines under key 'braille'
    out_braille = results.get('braille', [])
    if out_braille:
        return "\n".join(out_braille)
    return ""


def build_conversation(image_path, braille_text, mode, detection_text=""):
    """Build a single conversation in Qwen3-VL format."""
    user_content = []

    if mode == "retinanet_only":
        # Text-only: no image
        prompt = PROMPT_RETINANET_ONLY.format(detection=detection_text)
        user_content.append({"type": "text", "text": prompt})
    elif mode == "hybrid":
        # Image + detection hint
        user_content.append({"type": "image", "image": f"file://{os.path.abspath(image_path)}"})
        prompt = PROMPT_HYBRID.format(detection=detection_text)
        user_content.append({"type": "text", "text": prompt})
    else:
        # Vision-only (original)
        user_content.append({"type": "image", "image": f"file://{os.path.abspath(image_path)}"})
        user_content.append({"type": "text", "text": PROMPT_VISION_ONLY})

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": braille_text}]},
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for train.json and val.json")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--lang", type=str, default="EN")
    parser.add_argument("--mode", type=str, default="hybrid",
                        choices=["vision_only", "hybrid", "retinanet_only"],
                        help="Data preparation mode")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for RetinaNet inference (hybrid/retinanet_only modes)")
    parser.add_argument("--use_gt_as_detection", action="store_true",
                        help="Use ground-truth annotations as detection hint instead of running RetinaNet")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_root, "vision_llm")

    samples = collect_samples(args.data_root)
    print(f"Found {len(samples)} samples")
    print(f"Mode: {args.mode}")

    # Load RetinaNet if needed for hybrid/retinanet_only modes
    inferencer = None
    need_retinanet = args.mode in ("hybrid", "retinanet_only") and not args.use_gt_as_detection
    if need_retinanet:
        print(f"Loading RetinaNet model on {args.device}...")
        from model.infer_retinanet import BrailleInference
        inferencer = BrailleInference(device=args.device, verbose=0)
        print("RetinaNet loaded.")

    conversations = []
    skipped = 0
    for i, sample in enumerate(samples):
        try:
            # Read ground-truth annotations
            if sample["format"] == "json":
                rects, w, h = _read_labelme_annotations(sample["label"])
            else:
                img = PIL.Image.open(sample["image"])
                w, h = img.size
                rects = _read_dsbi_annotations(sample["label"], w, h)

            braille_text = rects_to_braille_text(rects, lang=args.lang)
            if not braille_text.strip():
                skipped += 1
                continue

            # Get detection text for hybrid/retinanet_only modes
            detection_text = ""
            if args.mode in ("hybrid", "retinanet_only"):
                if args.use_gt_as_detection:
                    # Use ground-truth as detection hint (simpler, no RetinaNet needed)
                    detection_text = braille_text
                else:
                    # Run RetinaNet inference
                    detection_text = run_retinanet_on_image(
                        inferencer, sample["image"], args.lang
                    )
                    if not detection_text.strip():
                        # RetinaNet found nothing — fall back to GT
                        detection_text = braille_text

            conv = build_conversation(
                sample["image"], braille_text, args.mode, detection_text
            )
            conversations.append(conv)
        except Exception as e:
            print(f"  Skipping {sample['image']}: {e}")
            skipped += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(samples)}")

    print(f"Total conversations: {len(conversations)}, skipped: {skipped}")

    # Deterministic split
    conversations.sort(key=lambda c: c["messages"][0]["content"][-1]["text"])
    n_val = max(1, int(len(conversations) * args.val_ratio))
    val_data = conversations[:n_val]
    train_data = conversations[n_val:]

    os.makedirs(args.output_dir, exist_ok=True)

    # Name output files based on mode
    suffix = f"_{args.mode}" if args.mode != "vision_only" else ""
    train_path = os.path.join(args.output_dir, f"train{suffix}.json")
    val_path = os.path.join(args.output_dir, f"val{suffix}.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print(f"Train: {len(train_data)} samples → {train_path}")
    print(f"Val:   {len(val_data)} samples → {val_path}")


if __name__ == "__main__":
    main()
