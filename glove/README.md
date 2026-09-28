# Glove subsystem (Karan)

Arduino Nano 33 BLE Sense Rev2 + 5 flex sensors. See `docs/PROJECT_PLAN.md` §3 for parts.

## Build / flash
1. Install VS Code + the PlatformIO extension.
2. Open `glove/firmware/` as a PlatformIO project.
3. Connect the Nano over micro-USB, click Upload, then open the Serial Monitor at 115200.
   You should see a CSV header followed by 50 rows/second.

## Wiring (per finger)
```
3.3V ──[flex sensor]──┬── A<n>
                      └──[10 kΩ]── GND
```
Mount the sensor's base rigidly on the back of the glove (hot glue + a fabric strip); only the strip should bend. Do not bend at the solder tabs.

## Data collection
```bash
python glove/tools/log_serial.py --port /dev/ttyACM0 --label OPEN --signer karan --seconds 3
```
Start with ~10 handshape classes (OPEN, FIST, POINT, etc.), 20 clips each, then calibrate: record `OPEN` and `FIST` once per session and store min/max per finger for normalization.

## Roadmap
- Wk 3–4: Edge Impulse project, first handshape classifier, compare flex-only / IMU-only / both
- Wk 5: on-device inference (Edge Impulse Arduino library), BLE broadcast of `{class_id, confidence, seq}`
- Wk 8: fusion with the camera model on the Pi (`pi/run.py`, late fusion of probabilities)
