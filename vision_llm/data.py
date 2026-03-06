"""Dataset for Braille VLM training.

Loads image + annotation pairs from AngelinaDataset (LabelMe JSON) and DSBI
(txt annotations), assembles ground-truth braille unicode text from the
per-character bounding box annotations using the existing postprocessing
pipeline (reading-order line assembly + braille interpretation).
"""

import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import PIL.Image
import torch
from torch.utils.data import Dataset

# Use existing AngelinaReader utilities (lazy imports to avoid liblouis dependency)
import braille_utils.label_tools as lt


def _read_labelme_annotations(json_path: str) -> List[Tuple]:
    """Read LabelMe JSON → list of (left, top, right, bottom, int_label) in pixel coords."""
    with open(json_path, "r", encoding="cp1251") as f:
        data = json.load(f)
    w, h = data["imageWidth"], data["imageHeight"]
    rects = []
    for shape in data["shapes"]:
        xs = [p[0] for p in shape["points"]]
        ys = [p[1] for p in shape["points"]]
        label_int = lt.human_label_to_int(shape["label"])
        if label_int is None or label_int == 0:
            continue
        rects.append((min(xs), min(ys), max(xs), max(ys), label_int))
    return rects, w, h


def _read_dsbi_annotations(txt_path: str, img_w: int, img_h: int) -> List[Tuple]:
    """Read DSBI txt annotation → list of (left, top, right, bottom, int_label) in pixel coords."""
    from data_utils.dsbi import read_DSBI_annotation
    # DSBI returns normalized [0,1) coords
    rects_norm = read_DSBI_annotation(txt_path, img_w, img_h, rect_margin=0.3, get_points=False)
    # Convert to pixel coords
    rects = []
    for r in rects_norm:
        rects.append((r[0] * img_w, r[1] * img_h, r[2] * img_w, r[3] * img_h, r[4]))
    return rects


def rects_to_braille_text(rects: List[Tuple], lang: str = "EN") -> str:
    """Convert bounding box annotations to braille unicode text.

    Uses the existing postprocess pipeline: boxes_to_lines assembles
    characters into reading-order lines, then we extract unicode braille.
    """
    if not rects:
        return ""
    from braille_utils.postprocess import boxes_to_lines
    boxes = [(r[0], r[1], r[2], r[3]) for r in rects]
    labels = [r[4] for r in rects]
    lines = boxes_to_lines(boxes, labels, lang=lang, filter_lonely=True)

    text_lines = []
    for ln in lines:
        line_str = ""
        for ch in ln.chars:
            line_str += lt.int_to_unicode(0) * ch.spaces_before
            line_str += lt.int_to_unicode(ch.label)
        text_lines.append(line_str)
    return "\n".join(text_lines)


def collect_samples(data_root: str) -> List[dict]:
    """Scan AngelinaDataset and DSBI for image+annotation pairs.

    Returns list of dicts: {"image": abs_path, "label": abs_path, "format": "json"|"txt"}
    """
    samples = []
    data_root = Path(data_root)

    # AngelinaDataset: LabelMe JSON annotations
    # Images are stored as *.labeled.jpg (the raw images aren't kept separately)
    angelina_dir = data_root / "AngelinaDataset"
    if angelina_dir.exists():
        for json_path in sorted(angelina_dir.rglob("*.labeled.json")):
            img_stem = str(json_path).replace(".labeled.json", "")
            # Try raw image first, then labeled.jpg (AngelinaDataset stores images as .labeled.jpg)
            found = False
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                img_path = img_stem + ext
                if os.path.isfile(img_path):
                    samples.append({"image": img_path, "label": str(json_path), "format": "json"})
                    found = True
                    break
            if not found:
                labeled_img = img_stem + ".labeled.jpg"
                if os.path.isfile(labeled_img):
                    samples.append({"image": labeled_img, "label": str(json_path), "format": "json"})

    # DSBI dataset: txt annotations
    dsbi_dir = data_root / "DSBI"
    for split_file in ["test.txt"]:  # train.txt has fewer, test.txt has 88
        split_path = dsbi_dir / split_file
        if not split_path.exists():
            continue
        with open(split_path) as f:
            for line in f:
                fn = line.strip().replace("\\", "/")
                if not fn:
                    continue
                img_path = dsbi_dir / fn
                # DSBI images might be in data/ subdir
                if not img_path.exists():
                    img_path = dsbi_dir / "data" / fn
                if not img_path.exists():
                    continue
                # Try recto variant
                recto = str(img_path).rsplit(".", 1)[0] + "+recto.jpg"
                if os.path.isfile(recto):
                    txt_label = recto.rsplit(".", 1)[0] + ".txt"
                    if os.path.isfile(txt_label):
                        samples.append({
                            "image": recto,
                            "label": txt_label,
                            "format": "txt",
                        })
                        continue
                txt_label = str(img_path).rsplit(".", 1)[0] + ".txt"
                if os.path.isfile(txt_label):
                    samples.append({
                        "image": str(img_path),
                        "label": txt_label,
                        "format": "txt",
                    })

    # Also include DSBI train.txt
    train_path = dsbi_dir / "train.txt"
    if train_path.exists():
        with open(train_path) as f:
            for line in f:
                fn = line.strip().replace("\\", "/")
                if not fn:
                    continue
                img_path = dsbi_dir / fn
                if not img_path.exists():
                    img_path = dsbi_dir / "data" / fn
                if not img_path.exists():
                    continue
                recto = str(img_path).rsplit(".", 1)[0] + "+recto.jpg"
                if os.path.isfile(recto):
                    txt_label = recto.rsplit(".", 1)[0] + ".txt"
                    if os.path.isfile(txt_label):
                        samples.append({
                            "image": recto,
                            "label": txt_label,
                            "format": "txt",
                        })
                        continue
                txt_label = str(img_path).rsplit(".", 1)[0] + ".txt"
                if os.path.isfile(txt_label):
                    samples.append({
                        "image": str(img_path),
                        "label": txt_label,
                        "format": "txt",
                    })

    # Deduplicate by image path
    seen = set()
    unique = []
    for s in samples:
        if s["image"] not in seen:
            seen.add(s["image"])
            unique.append(s)
    return unique


class BrailleVLMDataset(Dataset):
    """Dataset yielding (image_tensor, prompt, braille_text) for VLM training.

    image_tensor: (3, H, W) normalized float tensor
    prompt: the user instruction string
    braille_text: ground-truth braille unicode transcription
    """

    PROMPT = "Transcribe the braille in this image."

    def __init__(
        self,
        data_root: str,
        image_size: int = 1024,
        lang: str = "EN",
        split: str = "train",
        val_ratio: float = 0.1,
    ):
        self.image_size = image_size
        self.lang = lang

        all_samples = collect_samples(data_root)
        assert len(all_samples) > 0, f"No samples found in {data_root}"

        # Deterministic split
        all_samples.sort(key=lambda s: s["image"])
        n_val = max(1, int(len(all_samples) * val_ratio))
        if split == "val":
            self.samples = all_samples[:n_val]
        else:
            self.samples = all_samples[n_val:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load and preprocess image
        img = PIL.Image.open(sample["image"]).convert("RGB")
        img = img.resize((self.image_size, self.image_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        # Grayscale-normalize (matches AngelinaReader preprocessing)
        mean = arr.mean()
        std = max(arr.std(), 1e-6)
        arr = (arr - mean) / std
        image_tensor = torch.from_numpy(arr.transpose(2, 0, 1))  # (3, H, W)

        # Load annotations and convert to braille text
        if sample["format"] == "json":
            rects, w, h = _read_labelme_annotations(sample["label"])
        else:
            # For DSBI, need image dimensions
            orig_img = PIL.Image.open(sample["image"])
            w, h = orig_img.size
            rects = _read_dsbi_annotations(sample["label"], w, h)

        braille_text = rects_to_braille_text(rects, lang=self.lang)

        return {
            "image": image_tensor,
            "prompt": self.PROMPT,
            "target": braille_text,
            "path": sample["image"],
        }
