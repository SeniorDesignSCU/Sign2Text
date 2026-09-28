# Raspberry Pi 5 setup

Target: Raspberry Pi OS 64-bit (Bookworm), Pi 5 8 GB, Camera Module 3 Wide.

## 1. OS
Flash Raspberry Pi OS (64-bit) with Raspberry Pi Imager. Enable SSH and Wi-Fi in the imager's settings. Boot, then:

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y python3-picamera2 python3-opencv python3-venv git alsa-utils libatlas-base-dev
```

## 2. Project
```bash
git clone https://github.com/SeniorDesignSCU/Sign2Text.git ~/Sign2Text
cd ~/Sign2Text
python3 -m venv --system-site-packages .venv     # keeps apt's picamera2/opencv visible
source .venv/bin/activate
pip install -r requirements-pi.txt
bash scripts/download_models.sh
```

## 3. Camera check
```bash
rpicam-hello -t 5000            # should show a preview
python scripts/collect.py --signer test --session pi --camera 0   # if using a USB webcam
```

## 4. Piper TTS (offline)
```bash
pip install piper-tts
# voice model (~60 MB). Other voices: https://github.com/rhasspy/piper/blob/master/VOICES.md
cd data/models
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
cd ../..
echo "hello world" | piper --model data/models/en_US-lessac-medium.onnx --output-raw | aplay -r 22050 -f S16_LE -t raw -
```
If using the MAX98357A I2S amp instead of a USB speaker, add `dtoverlay=hifiberry-dac` to `/boot/firmware/config.txt` and reboot; `aplay -l` should then list it.

## 5. Run
```bash
python pi/run.py            # Pi camera, preview window
python pi/run.py --headless # for systemd
```

## 6. Boot-to-app (Week 14)
```bash
sudo cp pi/signs2speech.service /etc/systemd/system/
sudo systemctl enable --now signs2speech
journalctl -u signs2speech -f
```

## Performance notes
- Start at 640x480. MediaPipe hand+pose on the Pi 5 CPU is ~10–20 FPS; the model window is resampled so FPS variation doesn't break inference.
- If FPS < 10 after tuning: try hands-only (`LandmarkExtractor(use_pose=False)`) for a quick check, then consider the AI HAT+ (Week 4 decision).
- Keep the Active Cooler on; the Pi throttles at 80 °C.
