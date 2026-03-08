"""Prepare training data for Qwen3-VL fine-tuning.

Converts AngelinaDataset + DSBI annotations into Qwen3-VL chat format JSON.
Run from the AngelinaReader root directory with PYTHONPATH set.

Usage:
    PYTHONPATH=. python vision_llm/prepare_qwen3vl_data.py \
        --data_root . --output vision_llm/qwen3vl_data.json
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision_llm.data import collect_samples, _read_labelme_annotations, _read_dsbi_annotations, rects_to_braille_text
import PIL.Image


def build_conversation(image_path: str, braille_text: str) -> dict:
    """Build a single conversation in Qwen3-VL format."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                    {"type": "text", "text": "Transcribe the braille in this image."},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": braille_text},
                ],
            },
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for train.json and val.json")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--lang", type=str, default="EN")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_root, "vision_llm")

    samples = collect_samples(args.data_root)
    print(f"Found {len(samples)} samples")

    conversations = []
    skipped = 0
    for i, sample in enumerate(samples):
        try:
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

            conv = build_conversation(sample["image"], braille_text)
            conversations.append(conv)
        except Exception as e:
            print(f"  Skipping {sample['image']}: {e}")
            skipped += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(samples)}")

    print(f"Total conversations: {len(conversations)}, skipped: {skipped}")

    # Deterministic split
    conversations.sort(key=lambda c: c["messages"][0]["content"][0]["image"])
    n_val = max(1, int(len(conversations) * args.val_ratio))
    val_data = conversations[:n_val]
    train_data = conversations[n_val:]

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.json")
    val_path = os.path.join(args.output_dir, "val.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print(f"Train: {len(train_data)} samples → {train_path}")
    print(f"Val:   {len(val_data)} samples → {val_path}")


if __name__ == "__main__":
    main()
