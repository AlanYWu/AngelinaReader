# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AngelinaReader is an Optical Braille Recognition (OBR) system that converts photographs of Braille text into plain text. It uses a RetinaNet object detection CNN to detect and classify individual Braille cells (64 classes: blank + 63 dot patterns), then assembles them into lines and translates to text. Published at ICCV 2021 (Ovodov 2021).

## Key Commands

### Setup
```bash
git clone --recursive https://github.com/IlyaOvodov/AngelinaReader.git
pip install -r requirements.txt
wget -O weights/model.t7 http://ovdv.ru/files/retina_chars_eced60.clr.008
```

### Running
```bash
python run_web_app.py                          # Web app at http://127.0.0.1:5000
python api_img2braille_server.py               # REST API at http://0.0.0.0:12306
python run_local.py --input <file> [-l LANG]   # CLI inference
python model/train.py                          # Training (uses PyTorch Ignite)
```

### Testing (manual, no automated test suite)
```bash
python api_img2braille_test.py --input input_5.jpg       # Direct inference test
python test_image_to_braille_example.py                   # API endpoint test
python api_img2braille_client.py                          # API client test
python braille_utils/label_tools.py                       # Label encoding self-test (asserts)
```

## Architecture

### ML Pipeline

**Model**: RetinaNet with ResNet-50 FPN backbone (`pytorch_retinanet/` submodule)
- Classification head: 4× Conv2d → 64 classes (Braille dot patterns)
- Localization head: 4× Conv2d → bounding box regression
- Focal Loss (alpha=0.25, gamma=2), `class_loss_scale=100` in production
- Anchors: areas [128, 288, 512], aspect ratio [0.5], NMS threshold 0.02

**Training** (`model/train.py` + `model/params.py`):
- PyTorch Ignite framework via `ovotools.ignite_tools`
- Config in `model/params.py` as `AttrDict` — device set to `'mps'` (Apple Silicon); use `'cuda:0'` for NVIDIA GPUs
- Input resolution: (416, 416) for training, (1024, 1024) for inference
- Datasets: DSBI (`DSBI/`) and AngelinaDataset (`AngelinaDataset/`) — both are git submodules
- Augmentations via albumentations in `data_utils/data.py` (resize, rotation, blur, flip with label transform)
- Validation every 100 epochs using Levenshtein distance (`model/validate_retinanet.py`)
- Cyclical Learning Rate scheduler (warmup=10, min_lr=1e-5, max_lr=2e-4, period=500)

**Inference** (`model/infer_retinanet.py` → `BrailleInference`):
1. Image preprocessed → grayscale-normalized to 3-channel
2. Up to 8 orientation variants tested (0°/90°/180°/270° × recto/verso)
3. Best orientation selected by character validity statistics
4. NMS + decode via `pytorch_retinanet/encoder.py`
5. Postprocessing: geometric line assembly → Braille-to-text translation (`braille_utils/postprocess.py`)
6. Optional homography alignment for page straightening

### Braille Translation (`braille_utils/`)
- `letters.py`: dot pattern → character mappings for RU/EN/EN2/DE/GR/LV/PL/UZ/UZL
- `postprocess.py`: line assembly from boxes, state-machine interpreter (digit mode, caps mode, fractions, math)
- `postprocess_liblouis.py`: Grade-2 English via Liblouis back-translation (requires compiled Liblouis)
- `label_tools.py`: conversions between int/binary/dot-number/unicode/ascii label formats

### Web App (`web_app/`)
- Flask 0.12 with flask_login, flask_wtf, flask_mobility
- `angelina_reader_core.py`: `AngelinaSolver` wraps `BrailleInference`, `User` model with SQLite
- Templates in `web_app/templates/`, mobile variants in `web_app/templates/m/`

### Data Pipeline (`data_utils/`)
- `BrailleDataset`/`BrailleSubDataset`: loads DSBI `.txt` annotations or LabelMe `.json`
- `ImagePreprocessor`: albumentations pipeline with Braille-aware flip label transforms (`label_hflip`/`label_vflip`)

## Key Files

- `weights/model.t7` — pretrained model (gitignored, must download)
- `weights/param.txt` — serialized AttrDict production params
- `model/params.py` — training hyperparameters and data config
- `local_config.py` — local paths (data_path, liblouis tables)
- `.gitmodules` — submodules: `pytorch_retinanet`, `src/ovotools`

## Training on macOS (Apple Silicon / MPS)

The following fixes were applied to get training running locally (Feb 2025):

1. **Removed `pdb.set_trace()`** from `model/train.py` (was on line 26)
2. **Set device to `'mps'`** in `model/params.py` (was `'cpu'`; use `'cuda:0'` on NVIDIA GPUs)
3. **Set `can_overwrite=True`** in `model/params.py` settings to allow reusing existing config hash
4. **Created DSBI data symlinks**: `DSBI/train.txt` references paths like `Massage/M+1.jpg` but images are in `DSBI/data/Massage/`. Symlinks were created in `DSBI/` pointing to `DSBI/data/` subdirectories:
   ```bash
   cd DSBI && for dir in data/*/; do ln -sf "data/$(basename "$dir")" "$(basename "$dir")"; done
   ```
5. **Fixed ignite 0.5.x compatibility**: Added `skip_unrolling=True` to all `ignite.metrics.Loss` calls in `model/train.py` (ignite 0.5.x validates tuple output lengths, which breaks with the mismatched `y_pred`/`y` tuple sizes)
6. **Fixed param.txt path in validation**: Changed `os.path.join(ctx.params.get_base_filename(), 'param.txt')` to `ctx.params.get_base_filename() + '.param.txt'` in `model/train.py` (params.save writes sibling file, not inside directory)
7. **Fixed MPS device detection** in `model/infer_retinanet.py`: The original code only checked for CUDA; added MPS support so it doesn't fall back to CPU

### Running training
```bash
conda activate angelina
PYTHONPATH=/path/to/AngelinaReader python model/train.py
```

### Monitoring
```bash
# Watch training logs
tail -f NN_results/dsbi_lay5_*/log.log

# TensorBoard
conda activate angelina && tensorboard --logdir NN_results/dsbi_lay5_*/tb_log --port 6006
```

### Training speed
- ~4 sec/epoch on Apple Silicon MPS (26 train images, batch_size=12, 3 batches)
- 100,000 epochs (full config) would take ~4.6 days
- First CLR cycle completes at epoch 500 (~33 min)

## Known Issues

- `requirements.txt` is incomplete: missing `ignite`, `Levenshtein`, `opencv-python`, `requests`
- Flask/Werkzeug versions are very old (0.12/0.14.1) and may conflict with modern Python
- Liblouis (for EN Grade-2) must be compiled from source — not a pip package

## Git Submodules

- `pytorch_retinanet/` — custom RetinaNet implementation (IlyaOvodov/pytorch-retinanet)
- `src/ovotools/` — training utilities, Ignite helpers, AttrDict params (installed editable via requirements.txt)
- `AngelinaDataset/` — 284 annotated Braille photos
- `DSBI/` — Double-Sided Braille Image dataset (114 images, 26 train / 88 test)
