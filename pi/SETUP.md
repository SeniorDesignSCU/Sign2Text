# Raspberry Pi 5 setup

Written for the team's actual parts: Raspberry Pi 5 16 GB, 27 W USB-C power supply, Active Cooler,
Camera Module 3 Wide + Pi 5 camera cable, official Pi 5 case, SanDisk Extreme 64 GB microSD,
Adafruit MAX98357A I2S amp, Logitech C270 webcam, USB power bank. Vision only, no glove.

Each step ends with a check. Don't move on until the check passes.

## 0. Before you start

**Still needed**
- **A speaker for the amp.** The MAX98357A makes no sound on its own. Use a 4 Ω or 8 Ω speaker
  rated about 3 W. Bare wires or a JST plug are fine, because the amp has a screw terminal.
- **Header pins soldered onto the amp**, plus 5 female-to-female jumper wires. The header and
  the terminal block come in the bag with the amp.
- **Either** a micro-HDMI to HDMI cable plus a monitor, keyboard and mouse (the Pi 5 has
  micro-HDMI ports, not full-size), **or** just a laptop. Everything here works over SSH, and
  you can see the camera preview through Raspberry Pi Connect (step 3).

**Power.** Use the 27 W supply on the bench. Most power banks can't provide 5 V at 5 A. On one,
the Pi boots with a power warning and limits its USB ports to 600 mA. That's fine for demos with
the CSI camera, but not for development.

## 1. Assemble (power unplugged)

1. **Cooler.** Peel the film off the Active Cooler's thermal pads. Push its two spring pins into
   the board's mounting holes until they click, and plug its cable into the 4-pin **FAN** header.
   The Pi has one fan header, so leave out the case's own fan and its small stick-on heatsink.
   The fan assembly unclips from the case frame. If the case lid doesn't sit flat over the
   cooler, run with the lid off for now.
2. **Camera.** The camera ships with a standard-width cable already attached, which doesn't fit
   the Pi 5. Note which way its contacts face, slide the camera connector's latch out, swap in
   the Adafruit 22-pin to 15-pin cable the same way round, and close the latch.
   The cable's narrow end goes into one of the Pi 5's two small connectors marked CAM/DISP. For each
   connector, lift the latch, push the cable in straight and fully, then press the latch down.
   If step 3 finds no camera, power off and flip the cable at the Pi end. Inserting it the wrong
   way round doesn't damage anything.
3. Peel the protective film off the camera lens.

## 2. Flash the SD card (on a laptop)

1. Install **Raspberry Pi Imager** from raspberrypi.com/software.
2. Choose **Raspberry Pi 5**, then **Raspberry Pi OS (64-bit)**, the default with desktop, then
   your SD card.
3. When Imager offers to customize settings, set:
   - hostname `s2s`
   - a username and password (write them down; the rest of this guide calls it `$USER`)
   - Wi-Fi
   - **enable SSH**

   Campus Wi-Fi that needs a university login usually won't work from here. Do first setup on
   home Wi-Fi or a phone hotspot.
4. Write the card, put it in the Pi, plug in the 27 W supply, and wait about 2 minutes on the
   first boot.

**Check:** `ssh $USER@s2s.local` from your laptop gets you a prompt. If `.local` doesn't resolve,
find the Pi's IP address in your router or hotspot's device list.

## 3. Update and check the camera

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y git python3-venv python3-picamera2 alsa-utils libportaudio2
sudo reboot
```

**Check:**
```bash
rpicam-hello --list-cameras       # must list: imx708_wide
```

To see live video, there are two options:
- **Monitor attached:** run `rpicam-hello -t 10000`.
- **Laptop only:** run `rpi-connect on && rpi-connect signin`, open the URL it prints, then use
  "Screen sharing" at connect.raspberrypi.com.

**C270 webcam (optional):** plug it into a blue USB port.
**Check:** `ls /dev/video*` shows new devices and `lsusb | grep -i logitech` finds it.

## 4. Install the project

```bash
git clone https://github.com/SeniorDesignSCU/Sign2Text.git ~/Sign2Text
cd ~/Sign2Text
python3 -m venv --system-site-packages .venv     # keeps apt's picamera2 + numpy visible
source .venv/bin/activate
pip install -r requirements-pi.txt
bash scripts/download_models.sh                  # MediaPipe hand + pose models (~13 MB)
```

**Check:**
```bash
python -c "import mediapipe, picamera2; from ai_edge_litert.interpreter import Interpreter; print('ok', mediapipe.__version__)"
```

Run `source ~/Sign2Text/.venv/bin/activate` in every new terminal before running project scripts.

## 5. Hardware check: camera, landmarks, FPS

```bash
python scripts/check_hw.py                 # preview window (monitor or Connect screen sharing)
python scripts/check_hw.py --headless      # over plain SSH: prints FPS every second for 20 s
python scripts/check_hw.py --hands-only    # FPS without the pose model, for comparison
python scripts/check_hw.py --camera 0      # same test on the C270
```

Stand about 1 m from the camera and raise both hands.

**Check:**
- The average is **10 FPS or more**, which is the project target.
- Hands and pose are detected.
- `throttled` shows `0x0`.

Write down the FPS numbers. They go in the weekly report and settle the AI HAT+ question from
the plan.

## 6. Audio: MAX98357A amp and Piper voice

**Wire the amp** (power off). Leave the amp's SD and GAIN pins unconnected. With the GPIO
header along the top edge and the USB ports on the right, pins 1 and 2 are at the left end, and
even-numbered pins are in the outer row. Run `pinout` on the Pi to see a diagram and confirm.

| Amp pin | Pi GPIO | Physical pin |
|---|---|---|
| Vin | 5 V | 2 |
| GND | GND | 6 |
| BCLK | GPIO 18 | 12 |
| LRC | GPIO 19 | 35 |
| DIN | GPIO 21 | 40 |

Connect the speaker to the amp's `+` and `−` screw terminals.

**Enable the amp.** Open the config file:
```bash
sudo nano /boot/firmware/config.txt
```
Put a `#` in front of `dtparam=audio=on` and add this line under it:
```
dtoverlay=max98357a
```
Save, then run `sudo reboot`.

**Check:** `aplay -l` lists a card for the amp. Its short name is the word right after
`card N:`, for example `card 2: MAX98357A [...]`.

**Make the amp the default output** so every program uses it. Create the ALSA config:
```bash
sudo nano /etc/asound.conf
```
Paste this, replacing `MAX98357A` if `aplay -l` showed a different short name:
```
pcm.!default {
    type plug
    slave.pcm "hw:CARD=MAX98357A,DEV=0"
}
ctl.!default {
    type hw
    card MAX98357A
}
```
The `plug` layer matters: the amp only accepts stereo, and Piper produces mono.

**Install the voice and test:**
```bash
cd ~/Sign2Text && source .venv/bin/activate
python -m piper.download_voices en_US-lessac-medium --download-dir data/models
python scripts/check_hw.py --speaker
```

**Check:** you hear the test sentence.
- **Too quiet:** add `--volume 2`.
- **Distorted:** use `--volume 0.6`.
- **No sound:** try `--audio-device plughw:CARD=MAX98357A,DEV=0` to bypass the config file, then
  recheck the wiring.

Before the speaker arrives, you can test with a monitor's speakers:
`python scripts/check_hw.py --speaker --audio-device default`.

## 7. Run the recognizer (after the first model is trained)

Train on a laptop (see the README), then copy `data/models/` to the Pi:
```bash
scp data/models/sign_classifier.tflite data/models/labels.txt data/models/model_config.json \
    $USER@s2s.local:~/Sign2Text/data/models/
```

Then run:
```bash
python pi/run.py              # preview window
python pi/run.py --headless   # SSH / no screen
```

## 8. Boot-to-app (Winter Week 14)

```bash
sudo cp pi/signs2speech@.service /etc/systemd/system/
sudo systemctl enable --now signs2speech@$USER
journalctl -u signs2speech@$USER -f
```

## Troubleshooting

- **`rpicam-hello --list-cameras` shows nothing.** Power off, reseat both ends of the ribbon, and
  flip the narrow end if needed. Make sure the latches are fully closed.
- **`numpy.dtype size changed` or a picamera2 import error.** The SD card is on the older
  Bookworm image, where pip's newer numpy clashes with apt's. Reflash with the current default
  Raspberry Pi OS (step 2).
- **`throttled` is not `0x0`.**
  - `0x50005` or similar means under-voltage: use the 27 W supply.
  - `0x80008` means thermal throttling: check that the cooler's fan cable is plugged in.
- **FPS below 10.** Try these in order:
  1. Confirm you're at 640x480 (the default).
  2. Compare with `--hands-only`.
  3. Close the desktop preview with `--headless`.
  4. Only then consider the AI HAT+ (plan §3, optional).
- **Webcam opens but the picture is wrong or stretched.** Recording must be 640x480 (4:3).
  `collect.py` refuses other aspect ratios on purpose, because the features normalize x and y
  separately.
