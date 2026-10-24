# Signs to Speech

Edge-processed American Sign Language recognition with spoken output.
Santa Clara University ECEN & CSEN Cross-Disciplinary Senior Design, 2026–27 — Aditya Thakrar & Karan Velayutham.

A Raspberry Pi 5 with a camera extracts hand + upper-body landmarks (MediaPipe), classifies
word-level ASL signs with a temporal model (TFLite), and speaks the result through a speaker
(Piper TTS) through a MAX98357A I2S amp. Vision only (the glove was dropped in October 2026).
Everything runs on-device; no cloud.

- **Objectives:** [docs/PROJECT_OBJECTIVES.md](docs/PROJECT_OBJECTIVES.md)
- **Plan, BOM, schedule:** [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)
- **Pi hardware bring-up:** [pi/SETUP.md](pi/SETUP.md)
- **Research & setup log (all findings, decisions, full walkthrough):** [docs/RESEARCH_AND_SETUP_LOG.md](docs/RESEARCH_AND_SETUP_LOG.md)

## Layout
```
common/s2s/      shared library: camera, landmark extraction, features, clip I/O, vocabulary
scripts/         check_hw.py (hardware test), collect.py (data collection), download_models.sh
training/        train.py -> data/models/sign_classifier.tflite
pi/              run.py device runtime, SETUP.md, systemd unit
data/            clips/ (recorded landmark sequences), models/ (downloaded + trained)
docs/            objectives, plan
```

## Getting started

Two tracks run in parallel. Neither blocks the other.

**On the Pi:** follow [pi/SETUP.md](pi/SETUP.md) to `scripts/check_hw.py` passing (camera,
landmarks at 10 FPS or more, speaker).

**On each laptop (data collection, the critical path):** use Python 3.10-3.13.
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash scripts/download_models.sh                        # Windows without bash: download the two URLs in the script into data/models/
python scripts/check_hw.py --camera 0                  # landmarks + FPS on the webcam
python scripts/collect.py --signer <you> --session s01
```
Clips are recorded at 640x480 (4:3) to match the Pi. Don't change that, because the features
normalize x and y separately.

Record about 40 clips per sign (`r` records five in a row), including plenty of `NO_SIGN`. Then:
```bash
python training/train.py
```
which writes `data/models/sign_classifier.tflite`, `labels.txt`, and a confusion matrix.

## Run on the device
See [pi/SETUP.md](pi/SETUP.md). `python pi/run.py` does camera → landmarks → sliding window → TFLite → debounce → speech.

## How this differs from single-frame fingerspelling systems
Per-frame landmarks are normalized to the **signer's shoulders**, not the hand's bounding box,
so sign *location* is preserved; velocities are appended so *motion* is preserved; a window of
frames is classified by a 1D-CNN + GRU rather than one frame by an MLP. That is what lets it
recognize dynamic word signs like THANK-YOU or SORRY instead of only static letters.

## Data policy
Landmark clips (`data/clips/**/*.npz`) are small (~50 KB each) but many; they are gitignored
for now. Decision in Week 2: Git LFS vs. a shared drive. Never commit raw video of testers.
