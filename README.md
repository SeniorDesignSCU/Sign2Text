# Signs to Speech

Edge-processed American Sign Language recognition with spoken output.
Santa Clara University ECEN Senior Design, 2026–27 — Aditya Thakrar & Karan Velayutham.

A Raspberry Pi 5 with a camera extracts hand + upper-body landmarks (MediaPipe), classifies
word-level ASL signs with a temporal model (TFLite), and speaks the result through a speaker
(Piper TTS). An instrumented glove (Arduino Nano 33 BLE Sense Rev2) adds finger-flexion and
motion data for handshape-ambiguous signs. Everything runs on-device; no cloud.

- **Objectives:** [docs/PROJECT_OBJECTIVES.md](docs/PROJECT_OBJECTIVES.md)
- **Plan, BOM, schedule:** [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)

## Layout
```
common/s2s/      shared library: landmark extraction, features, clip I/O, vocabulary
scripts/         collect.py (data collection), download_models.sh
training/        train.py -> data/models/sign_classifier.tflite
pi/              run.py device runtime, SETUP.md, systemd unit
glove/           Nano 33 BLE Sense firmware (PlatformIO) + serial logger
data/            clips/ (recorded landmark sequences), models/ (downloaded + trained)
docs/            objectives, plan
```

## Quick start (laptop, Week 1)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash scripts/download_models.sh
python scripts/collect.py --signer <you> --session s01
```
Record ~40 clips per sign (`r` records five in a row), including plenty of `NO_SIGN`. Then:
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
