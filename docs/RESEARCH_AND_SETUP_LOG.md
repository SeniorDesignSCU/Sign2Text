# Signs to Speech — Research & Setup Log

**Purpose:** the single place for everything decided, researched, verified and built so far, so
nobody has to dig through old chat threads. If you start a new chat with Claude, paste or attach
this file first. It's the full context.

**Last updated:** Saturday, Oct 24, 2026 · **Repo:** https://github.com/SeniorDesignSCU/Sign2Text
**Commit at time of writing:** `c74575d` (vision-only bring-up) on `main`

---

## Contents

1. [Project snapshot](#1-project-snapshot)
2. [Decision history](#2-decision-history)
3. [Hardware: what we own, what's still needed](#3-hardware-what-we-own-whats-still-needed)
4. [Software stack, verified Oct 24, 2026](#4-software-stack-verified-oct-24-2026)
5. [Findings and bugs found during research](#5-findings-and-bugs-found-during-research)
6. [What was tested, and what wasn't](#6-what-was-tested-and-what-wasnt)
7. [Hardware setup, full walkthrough](#7-hardware-setup-full-walkthrough)
8. [Laptop setup and data collection](#8-laptop-setup-and-data-collection)
9. [Training and deploying a model](#9-training-and-deploying-a-model)
10. [Repo map and command cheat sheet](#10-repo-map-and-command-cheat-sheet)
11. [Schedule status and course deadlines](#11-schedule-status-and-course-deadlines)
12. [Open items and next steps](#12-open-items-and-next-steps)
13. [Troubleshooting reference](#13-troubleshooting-reference)
14. [Reference data: model files, pinout, links](#14-reference-data-model-files-pinout-links)
15. [Earlier chat threads](#15-earlier-chat-threads)

---

## 1. Project snapshot

| | |
|---|---|
| **Name** | Signs to Speech (repo: Sign2Text) |
| **What it does** | Recognizes isolated ASL word/phrase signs from a camera and speaks them aloud. Everything runs on the device, with no cloud. |
| **Team** | Aditya Thakrar (ECE) and Karan Velayutham (CSE: AI/ML, programming) |
| **Course** | SCU Senior Design (ECEN 192, Senior Design 1), cross-disciplinary ECEN & CSEN, 2026–27 |
| **Advisors** | Prof. Michael Schimpf, Fr. Dat Tran, S.J. |
| **Instructor** | Sahana |
| **Timeline** | Started Sept 22, 2026. Fall and Winter quarters. Working product Feb 28, 2027; final demo and report by Mar 12, 2027. |
| **Budget** | Self-funded by the students, under $1,000 total |
| **Platform** | Raspberry Pi 5 (16 GB) + Camera Module 3 Wide. MediaPipe landmarks, a temporal model run with LiteRT (the TFLite runtime), Piper TTS, and a MAX98357A I2S amp driving a speaker. |
| **Scope** | **Vision only.** The glove was dropped in October 2026. |

**Measurable targets** (from `docs/PROJECT_OBJECTIVES.md`):

| Metric | Target |
|---|---|
| Vocabulary | 25 or more word/phrase signs (plus A–Z fingerspelling as a secondary mode) |
| Top-1 accuracy, team signers | 90% or more |
| Top-1 accuracy, unseen signers | 80% or more |
| Latency, end of sign to start of speech | 1 s or less |
| Landmark throughput on the device | 10 FPS or more |
| User testing | 5 or more participants, at least 1 fluent signer |
| Form factor | Standalone, battery-powered, boots straight into the app |

**Explicit non-goals:** full ASL grammar or translation, facial-expression grammar, continuous
signing without pauses.

**Starter vocabulary** (`common/s2s/vocab.py`; revise with an ASL-fluent reviewer):
HELLO, THANK_YOU, PLEASE, SORRY, YES, NO, HELP, NAME, WHAT, WHERE, WANT, NEED, EAT, DRINK, WATER,
BATHROOM, MORE, STOP, GOOD, BAD, ME, YOU, LOVE, FRIEND, HOME, WORK, LEARN, TIME. There are also
two control signs, CLEAR and SPEAK, which are recognized but not spoken, and the required
`NO_SIGN` idle class.

---

## 2. Decision history

| When | Decision | Why |
|---|---|---|
| Sept 2026 | **Raspberry Pi 5 + camera instead of Arduino Portenta H7 + Vision Shield** | The Portenta is a microcontroller. It can't run MediaPipe, and its camera only supports roughly 96×96 grayscale pixel classification. Word signs are dynamic, often two-handed, and depend on where the hands are relative to the body. That needs hand and pose landmarks, which needs a full OS. The Pi 5 still counts as edge processing. |
| Sept 2026 | **MVP = word/phrase signs**, not fingerspelling only | That's the useful version for a Deaf user talking to a hearing person. Fingerspelling is a late secondary mode. |
| Sept 2026 | **Output = speech** (text is a free byproduct) | Speech is what makes it a standalone device rather than a laptop demo. It also adds a real subsystem: offline TTS, audio hardware and a latency budget. |
| Sept 2026 | Build on the *idea* of the reference repo (Muhib-Mehdi/ASL-Recognition-System), not its code | That repo does static A–Z on a laptop. We add dynamic word signs, pose landmarks, a temporal model, edge deployment and TTS. Those are exactly the items its own roadmap lists as unfinished. |
| Oct 2026 | **Glove dropped. Vision only.** | Team decision, confirmed: "we have no plan on doing a glove at all." Two-board sensor fusion was the riskiest part of the schedule. |
| Oct 24, 2026 | **Raspberry Pi OS 64-bit, current default (Debian 13 "Trixie", Python 3.13)**, not the older Bookworm | Verified that MediaPipe 1.1.1 and every other dependency installs on Python 3.13. On Bookworm, pip's numpy clashes with the apt-installed picamera2. Details in §5. |
| Oct 24, 2026 | **Record and run everything at 640×480 (4:3)** | The features normalize x and y against the image width and height separately, so the aspect ratio must match between training and the Pi. Details in §5. |
| Oct 24, 2026 | **ai-edge-litert instead of tflite-runtime** on the Pi | tflite-runtime is abandoned: the last release was 2.14 in Oct 2023, with no Python 3.12 or 3.13 wheels. |
| Oct 24, 2026 | **TensorFlow pinned to 2.20.x** for training on laptops | `tf.lite` (the converter) is deprecated from 2.20 and slated for removal. 2.20 still has it and ships for Python 3.10–3.13 on Windows, Linux and Apple-silicon Macs. |

---

## 3. Hardware: what we own, what's still needed

### Owned

| Item | SKU / ID | Notes |
|---|---|---|
| Raspberry Pi 5, 16 GB | PI5-16GB | More RAM than the plan's 8 GB, which is fine |
| Raspberry Pi 5 Power Supply 27 W USB-C (black) | DCAR-RSP-PI-PD-BLK | The only supply that gives the Pi 5 full USB current (5 V / 5 A) |
| Raspberry Pi 5 Active Cooler | RSP-ACTIVE-COOLER | Needed for sustained ML; without it the Pi throttles at 80 °C |
| Raspberry Pi Camera Module 3 – Wide | RSP-CAM-V3-WIDE | Sensor IMX708, 120° field of view, autofocus. Ships with a standard-width cable that doesn't fit the Pi 5. |
| Raspberry Pi 5 FPC camera cable, 22-pin 0.5 mm to 15-pin 1 mm, 200 mm | Adafruit 5818 ($2.70) | Narrow end to the Pi 5, wide end to the camera |
| Raspberry Pi 5 Case (white/red) | RSP-CASE5-PI5-WHT | Has its own fan and small heatsink. **Don't use them with the Active Cooler** (see §5). |
| SanDisk Extreme 64 GB microSD | — | The OS card |
| Adafruit I2S 3 W Class D Amplifier, MAX98357A | Adafruit 3006 ($5.95) | Header and screw terminal come loose and need soldering |
| Logitech C270 HD webcam | — | 720p USB. Used for laptop data collection, and as a backup camera on the Pi. |
| USB power bank | Amazon | For demos only. Most banks can't supply 5 V / 5 A (see §5). |

### Still needed

| Item | Why | Rough cost |
|---|---|---|
| **Speaker, 4 Ω or 8 Ω, about 3 W** | The amp produces no sound without one. Adafruit sells suitable enclosed 3 W 4 Ω speakers. | $3–8 |
| **Female-to-female jumper wires (5)** | To connect the amp's header pins to the Pi's GPIO pins | $3 |
| Soldering iron access | To solder the header and screw terminal onto the amp (SCU lab) | — |
| *Optional:* micro-HDMI to HDMI cable | The Pi 5 has micro-HDMI ports only. Not needed if you work over SSH and Raspberry Pi Connect. | $8 |
| *Optional:* camera mount or small tripod | To hold the camera steady at chest height about 1 m from the signer | $10 |
| *Optional, later:* Raspberry Pi AI HAT+ | Only if the measured FPS stays under 10 after tuning | $70–110 |

---

## 4. Software stack, verified Oct 24, 2026

These versions were checked against PyPI and tested in a sandbox on Oct 24, 2026.

| Component | Version | Where | Notes |
|---|---|---|---|
| Raspberry Pi OS | 64-bit, Trixie (Debian 13), Python 3.13 | Pi | The Raspberry Pi Imager default. Includes picamera2 and Raspberry Pi Connect. |
| MediaPipe | **1.1.1** (released Oct 14, 2026) | Pi + laptops | Wheel is `py3-none-manylinux_2_28_aarch64`, so it works with any Python 3, including 3.13. The Tasks API we use (`HandLandmarker`, `PoseLandmarker`, `RunningMode`, `mp.Image`) is unchanged. It pulls in `opencv-contrib-python`, `sounddevice` and `matplotlib`. |
| ai-edge-litert | **2.3.0** (Oct 6, 2026) | Pi | The renamed TFLite interpreter, with Python 3.10–3.14 aarch64 wheels. Import with `from ai_edge_litert.interpreter import Interpreter`. |
| tflite-runtime | 2.14.0 (Oct 2023), **abandoned** | — | No wheels for Python 3.12 or newer, so it won't install on current Pi OS. Removed from requirements. |
| piper-tts | **1.8.0** (Sept 4, 2026) | Pi | The command still supports `--model`, `--output-raw` and `--volume`, and reads text from stdin. Download a voice with `python -m piper.download_voices <voice> --download-dir <dir>`. |
| TensorFlow | **2.20.x** pinned (2.21.0 is latest) | Laptops | `tf.lite` is deprecated from 2.20 and will be removed. 2.20 ships for Python 3.10–3.13, Windows, Linux and Apple-silicon macOS. **No Intel-Mac wheel.** |
| picamera2 | apt package `python3-picamera2` | Pi | Must come from apt and be used through a venv created with `--system-site-packages`. |
| OpenCV | `opencv-contrib-python` (pulled in by MediaPipe) | Pi + laptops | Don't also install `opencv-python`. Both provide `cv2` and they overwrite each other. |

**Outdated advice you'll find online:** several guides (e.g., SunFounder's MediaPipe setup page)
say MediaPipe doesn't support Python 3.13 and tell you to use Bookworm. That was true for
MediaPipe 0.10.x. It is no longer true for 1.1.x.

---

## 5. Findings and bugs found during research

Each of these would have cost real time during bring-up. All are fixed in commit `c74575d`.

1. **`cv2.VideoCapture(0)` can't open the Pi Camera Module on a Pi 5.**
   - The CSI camera is a libcamera device, so it needs picamera2.
   - The old `collect.py` only used `cv2.VideoCapture`, so it couldn't record on the Pi.
   - **Fix:** `common/s2s/camera.py` opens the right source for every script. Use `--camera auto`
     (the default), `pi`, a webcam index, or a video file path.
2. **Color swap with the Pi camera.**
   - picamera2's `"RGB888"` format is stored as B, G, R in memory, which is already OpenCV's BGR.
   - The old `run.py` converted it from RGB to BGR, which swapped red and blue before MediaPipe
     saw the frame.
   - **Fix:** no conversion for picamera2 frames.
3. **Aspect ratio mismatch would quietly hurt accuracy.**
   - MediaPipe returns x and y normalized to the image's width and height separately.
   - `features.py` re-centers on the shoulders and scales by shoulder width, but it can't undo a
     different aspect ratio.
   - So a clip recorded at 16:9 (e.g., the old `collect.py` default of 960×540) looks
     horizontally squashed compared to the Pi's 4:3.
   - **Fix:**
     - All scripts default to 640×480.
     - `collect.py` refuses a camera that isn't 4:3 unless you pass `--allow-any-aspect`.
     - Each clip now stores its `width` and `height`.
     - `train.py` warns about non-4:3 clips.
4. **tflite-runtime no longer installs on current Pi OS.** It was replaced with ai-edge-litert.
   `run.py` tries `ai_edge_litert`, then `tflite_runtime`, then `tensorflow`.
5. **The old TensorFlow pin `<2.20` had no Python 3.13 build.**
   - TF 2.19 only ships for Python 3.9–3.12.
   - **Fix:** pinned to `>=2.20,<2.21`.
   - `train.py` now tries the direct Keras conversion first, then falls back to exporting a
     SavedModel and converting that. Some Keras 3 models fail the direct path.
6. **`opencv-python` and `opencv-contrib-python` clash.** MediaPipe 1.x depends on the contrib
   build, so `opencv-python` was removed from `requirements.txt`.
7. **numpy clash on Bookworm.**
   - On the older Pi OS, apt's picamera2 is built against numpy 1.24, and pip can install
     numpy 2.x into the venv.
   - The result is `numpy.dtype size changed` errors.
   - On Trixie, apt's numpy is already 2.x, so there's no clash. Another reason to use the
     current default image.
8. **Case fan vs. Active Cooler.**
   - The official Pi 5 case has its own fan, which plugs into the Pi's **single** 4-pin FAN
     header, plus a small stick-on heatsink.
   - With the Active Cooler installed, leave the case fan and heatsink out. The fan assembly
     unclips from the case's white frame.
   - If the lid doesn't sit flat over the cooler, run lid-off on the bench.
9. **Power bank limits.**
   - The Pi 5 wants 5 V at 5 A for full USB current.
   - On a supply that only negotiates 3 A, it warns at boot and caps USB peripherals at 600 mA
     total.
   - The CSI camera doesn't use USB, so demos on a power bank are fine. Use the 27 W supply for
     development.
10. **MAX98357A specifics.**
    - The kernel doesn't support mono I2S output, and Piper produces mono, 22,050 Hz audio.
    - So audio must go through ALSA's `plug` layer: either `/etc/asound.conf` pointing `default`
      at `plughw:` of the amp, or `aplay -D plughw:...`.
    - The amp has no volume control in software. Use Piper's `--volume` (exposed as
      `run.py --volume` and `check_hw.py --volume`).
11. **Camera Module 3 comes with the wrong cable for a Pi 5.** It ships with a 15-pin to 15-pin
    cable for older Pis. Swap in the 22-pin to 15-pin cable (Adafruit 5818).
12. **No bring-up path existed.**
    - `run.py` needs a trained model, so before this there was nothing to test hardware with.
    - **Fix:** `scripts/check_hw.py` tests camera, landmarks, FPS and the speaker with no model.
13. **The glove code and dependencies are gone.** Removed `glove/`, `bleak` and `pyserial`. The
    systemd unit is now a per-user template (`signs2speech@.service`), because new Pi OS images
    have no default `pi` user.

---

## 6. What was tested, and what wasn't

**Tested in a sandbox** (x86 Linux, Python 3.13, MediaPipe 1.1.1, with the real `.task` model
files):
- `LandmarkExtractor` loads both models and processes frames. The features come out as
  `(T, 300)` and are finite.
- `check_hw.py --headless`:
  - Runs on a 640×480 video file and prints FPS and a PASS line.
  - Flags a 1280×720 video as not 4:3.
  - Gives a clear error when no camera exists.
- `collect.py`:
  - Refuses a 16:9 source.
  - Records a clip from a 4:3 source (keypresses simulated) with `source`, `width` and `height`
    saved.
  - `train.load_dataset()` reads that clip, including mirror augmentation.
- `run.py` gives a clear message when no model has been trained yet.
- `ai_edge_litert` and `piper-tts` 1.8 install on Python 3.13; Piper's command-line flags were
  confirmed.
- The pushed commit was re-tested from a fresh clone.

**Not tested yet:**
- **Anything on the physical Pi:** picamera2 capture, real FPS, I2S audio, the overlay. You
  verify these with the checks in §7.
- **Real hand detection.** The sandbox test frames contained no hands. `check_hw.py` with you in
  front of the camera is the real test.
- **`train.py` end to end on TensorFlow 2.20,** including TFLite export. TensorFlow wasn't
  installed in the sandbox. The first laptop training run confirms it; if it fails, keep the
  full error message.
- **LiteRT running our trained `.tflite`** on the Pi. Confirmed once the first model exists.

---

## 7. Hardware setup, full walkthrough

`pi/SETUP.md` has the same steps in short form. This section explains each step in more depth.
**Every step ends with a check. Don't move on until it passes.**

### 7.1 Before you start
- Clear workspace, nothing metal under the board. Touch something grounded before handling the
  Pi to discharge static.
- **Everything in 7.2 happens with the power unplugged.**
- Have the 27 W supply, the microSD card and a laptop with an SD card reader.

### 7.2 Assemble

**A. Active Cooler**
1. Peel the clear protective film off the cooler's thermal pads, on the underside.
2. Line up the two spring-loaded push pins with the two mounting holes beside the processor.
   The cooler's fan cable should point toward the FAN header.
3. Push both pins until they click. The pads should press flat on the chips.
4. Plug the cooler's cable into the small 4-pin **FAN** connector on the board. It's keyed and
   only goes in one way, so don't force it.

**B. Camera cable**
1. **At the camera:** the Camera Module 3 comes with a cable already attached. Before removing
   it, **note which side its contacts face.**
   1. Gently pull the connector's latch outward about 1 mm. It slides; it doesn't come off.
   2. Pull the old cable out.
   3. Insert the **wide** end of the new cable the same way round.
   4. Push the latch back in.
2. **At the Pi:** the Pi 5 has two small connectors labeled **CAM/DISP 0** and **CAM/DISP 1**.
   Either works.
   1. Lift that connector's latch.
   2. Slide the **narrow** end of the cable in straight and fully.
   3. Press the latch down.
   4. If the camera isn't detected in 7.4, power off and flip the cable at this end. Getting it
      backwards doesn't damage anything; it just won't connect.
3. Peel the protective film off the camera lens.
4. Don't fold the ribbon sharply. Gentle curves only.

**C. Case** (optional for now)
1. Unclip the case's own fan assembly from the white frame and set it aside along with the
   case's small heatsink. The Active Cooler replaces both.
2. Seat the Pi in the base and route the camera ribbon out of the case.
3. If the lid doesn't close over the cooler, leave it off while developing.

**D. Amp** (solder now if you have access to an iron; wire it up later in 7.7)
1. Snap the header strip to length if needed. Insert its short side through the amp's pin holes
   from underneath. Plugging the long side into a breadboard holds it straight while soldering.
2. Solder each pin on the top side.
3. Solder the 2-pin screw terminal into the speaker `+` and `−` holes, with the screw openings
   facing outward.
4. Leave the **SD** and **GAIN** pins unconnected.
   - Unconnected SD means the amp outputs the left and right channels mixed together.
   - Unconnected GAIN means 9 dB gain.

### 7.3 Flash the OS (on your laptop)
1. Download **Raspberry Pi Imager** from raspberrypi.com/software and install it.
2. **Device:** Raspberry Pi 5.
3. **OS:** Raspberry Pi OS (64-bit), the default desktop version. Don't pick "Legacy" or "Lite".
   You want the desktop for screen sharing.
4. **Storage:** the SanDisk card. Double-check it's not a laptop drive.
5. When Imager offers to customize settings:
   - **Hostname:** `s2s`
   - **Username and password:** choose them and write them down. Both teammates can share one
     login.
   - **Wi-Fi:** SSID and password. Campus networks that need a university login (WPA2-Enterprise)
     usually can't be set here. Do first setup at home or on a phone hotspot, and add campus
     Wi-Fi later from the desktop's network menu.
   - **Locale:** time zone America/Los_Angeles, keyboard US.
   - **Services:** enable **SSH** with password login.
6. Write the card and wait for it to verify. Eject it, put it in the Pi's microSD slot on the
   underside, plug in the camera, then plug in the 27 W supply last.
7. The first boot takes about 2 minutes, and the Pi may reboot once while it resizes the
   filesystem.

**Check:** from the laptop, on the same network, `ssh <username>@s2s.local` gets you a prompt.
If `.local` doesn't resolve (common on Windows or some hotspots), find the Pi's IP address in
the router's or hotspot's device list and use `ssh <username>@<ip>`.

### 7.4 Update the system and check the camera
```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y git python3-venv python3-picamera2 alsa-utils libportaudio2
sudo reboot
```

**Check:**
```bash
rpicam-hello --list-cameras
```
It must list a camera with `imx708_wide`.

**To see live video:**
- **Monitor attached:** `rpicam-hello -t 10000` shows 10 seconds of preview.
- **Laptop only:**
  1. Run `rpi-connect on`, then `rpi-connect signin`.
  2. Open the printed URL and sign in with a Raspberry Pi ID (free).
  3. At connect.raspberrypi.com, choose your device, then **Screen sharing**.
  4. Run preview commands in a terminal on that shared desktop.

**Check the Pi's health:**
```bash
vcgencmd measure_temp      # idle should be roughly 40–55 °C with the cooler
vcgencmd get_throttled     # should print throttled=0x0
pinout                     # prints a GPIO diagram for this board (useful in 7.7)
```

**C270 webcam (optional):** plug it into a blue USB 3 port.
**Check:** `lsusb | grep -i logitech` finds it.

### 7.5 Install the project on the Pi
```bash
git clone https://github.com/SeniorDesignSCU/Sign2Text.git ~/Sign2Text
cd ~/Sign2Text
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements-pi.txt
bash scripts/download_models.sh
```

- `--system-site-packages` lets the venv use the apt copies of picamera2 and numpy, which pip
  can't install properly.
- `download_models.sh` fetches about 13 MB of MediaPipe models. Their hashes are in §14.

**Check:**
```bash
python -c "import mediapipe, picamera2; from ai_edge_litert.interpreter import Interpreter; print('ok', mediapipe.__version__)"
```
It should print `ok 1.1.1` or newer.

Every new terminal needs `cd ~/Sign2Text && source .venv/bin/activate` before running project
scripts.

### 7.6 Hardware check: camera, landmarks, FPS
```bash
python scripts/check_hw.py                 # live window with landmarks drawn (monitor or Connect)
python scripts/check_hw.py --headless      # plain SSH: prints FPS every second for 20 s
python scripts/check_hw.py --hands-only    # FPS without the pose model, for comparison
python scripts/check_hw.py --camera 0      # same test using the C270
```

Stand about 1 m away with both hands raised, facing the camera, in decent light.

**Check, and record these numbers for the weekly report:**
- **Average FPS:** 10 or more is a PASS and the project target. The plan estimated 10–20 FPS for
  hands plus pose on the Pi 5's processor.
- **Detection:** hands and pose are detected in most frames.
- **`throttled`:** stays `0x0` during the run.
- If FPS is under 10, try `--hands-only` and `--headless` to see where the time goes. That
  decides whether you need the AI HAT+ (plan §3).

### 7.7 Audio: amp wiring, enabling it, and the voice

**Wire it up with the power off.** Look at the board with the GPIO header along the top edge and
the USB and Ethernet ports on the right.
- **Pins 1 and 2** are at the **left** end, away from the USB ports.
- **Even-numbered pins** are in the outer row along the board edge; odd pins are in the inner
  row.
- If unsure, run `pinout` on the Pi first and match its picture.

| Amp pin | Connects to | Physical pin | Row / position |
|---|---|---|---|
| Vin | 5 V | **2** | outer row, 1st from left |
| GND | Ground | **6** | outer row, 3rd from left |
| BCLK | GPIO 18 (I2S clock) | **12** | outer row, 6th from left |
| LRC | GPIO 19 (I2S frame) | **35** | inner row, 18th from left |
| DIN | GPIO 21 (I2S data) | **40** | outer row, 20th (last, nearest the USB ports) |

Screw the speaker wires into the amp's `+` and `−` terminals. Polarity doesn't matter for a
single speaker.

**Enable the amp in the boot config:**
```bash
sudo nano /boot/firmware/config.txt
```
1. Find `dtparam=audio=on` and put a `#` in front of it.
2. Add this line below it:
   ```
   dtoverlay=max98357a
   ```
3. Save (Ctrl+O, Enter, Ctrl+X), then run `sudo reboot`.

**Check:** `aplay -l` lists a new card. The short name is the word right after `card N:`, for
example `card 2: MAX98357A [...]`.

**Make the amp the default output:**
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

**Install the voice and test speech:**
```bash
cd ~/Sign2Text && source .venv/bin/activate
python -m piper.download_voices en_US-lessac-medium --download-dir data/models
python scripts/check_hw.py --speaker
```

**Check:** you hear "Hello. This is Signs to Speech…"
- **Too quiet:** add `--volume 2`.
- **Distorted:** use `--volume 0.6`.
- **No sound:** try `--audio-device plughw:CARD=MAX98357A,DEV=0`. If that works, the
  `asound.conf` file is wrong. If it doesn't, recheck the wiring against `pinout`.
- **Before the speaker arrives:** test through a monitor's speakers with
  `--audio-device default`, before creating `asound.conf`.

### 7.8 Done when
- `rpicam-hello --list-cameras` finds `imx708_wide`.
- `check_hw.py` shows 10 FPS or more with hands and pose detected.
- `check_hw.py --speaker` is audible.
- `vcgencmd get_throttled` is `0x0`.

Put these numbers in the weekly status report as the "hardware bring-up complete" milestone.

---

## 8. Laptop setup and data collection

**Data collection is the critical path, and it doesn't need the Pi.** Both teammates can record
on laptops now, using the C270 or a built-in webcam.

### Setup
Use Python 3.10–3.13. On an Intel Mac, use Python 3.12 or older and expect TensorFlow trouble;
training can run on Google Colab instead.
```bash
git clone https://github.com/SeniorDesignSCU/Sign2Text.git && cd Sign2Text
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash scripts/download_models.sh
python scripts/check_hw.py --camera 0
```
- **Windows without bash:** download the two model URLs in §14 into `data/models/` by hand.
- `check_hw.py --camera 0` should show a landmark overlay on the webcam. Try `--camera 1` if the
  wrong camera opens.

### Recording
```bash
python scripts/collect.py --signer aditya --session s01          # or --signer karan
```
- **SPACE:** countdown, then record one clip.
- **r:** record 5 clips back to back.
- **n / p:** next or previous sign.
- **l:** cycle the lighting tag (bright / office / dim).
- **q:** quit.
- Clips are saved to `data/clips/<SIGN>/<signer>_<session>_<nnn>.npz`, about 50 KB each, and
  are gitignored.

### Protocol
- **Target:** about 40 clips per sign per signer in round 1, **plus lots of `NO_SIGN`**: idle
  hands, scratching your face, adjusting glasses. `NO_SIGN` is what stops the device from
  speaking random words.
- **Framing:** stand about 1 m from the camera, with the frame covering your head to your waist
  and both elbows visible.
- **Vary conditions across sessions:** lighting, background, clothing, distance (0.8–1.5 m), and
  seated vs. standing. Tag the lighting with `l`.
- **Sign correctly:** use Lifeprint / ASL University as the reference. Get an ASL-fluent person
  to review the vocabulary and your form early.
- Keep 640×480. If a webcam can't do 4:3, `collect.py` stops with a message.
- **Back up clips** (shared drive or Git LFS). The dataset is the most valuable artifact of the
  project. Decide which storage you'll use soon.

### Public data that can help
- **Google Isolated Sign Language Recognition (Kaggle, 2023):** MediaPipe landmark sequences for
  250 ASL signs, a close match to our representation. Good for pretraining and for studying the
  winning model designs. Check the license before citing it in the report.
- **WLASL:** word-level ASL video dataset, about 2,000 signs. Useful for vocabulary choices and
  extra clips run through our landmark extractor.

---

## 9. Training and deploying a model

On a laptop, after recording clips:
```bash
python training/train.py                    # defaults: window 40, 60 epochs, mirror augmentation
```
- **Validation split:** if two or more signers exist, it holds out one signer, which is the
  honest cross-signer metric. Otherwise it uses a random 80/20 split.
- **Outputs** in `data/models/`:
  - `sign_classifier.keras`
  - `sign_classifier.tflite`
  - `labels.txt`
  - `model_config.json`
  - `confusion_matrix.png`
- **Model:** 1D-CNN, then GRU, then a dense layer, on 300 features per frame. Each frame has
  left-hand, right-hand and pose positions plus their frame-to-frame velocities, normalized to
  the shoulders.

Copy the model to the Pi:
```bash
scp data/models/sign_classifier.tflite data/models/labels.txt data/models/model_config.json \
    <username>@s2s.local:~/Sign2Text/data/models/
```

Then run it on the Pi:
```bash
python pi/run.py                  # live window
python pi/run.py --headless       # no screen
```
- **Options:**
  - `--camera`
  - `--min-conf 0.8`
  - `--stable-frames 6`
  - `--audio-device`
  - `--volume`
  - `--no-tts`
- **Behavior:**
  - Each recognized word is spoken as soon as it's recognized.
  - SPEAK says the buffered sentence.
  - CLEAR empties the buffer.
  - A word isn't repeated until an idle gap of about 0.6 s.

Boot-to-app, scheduled for Winter Week 14:
```bash
sudo cp pi/signs2speech@.service /etc/systemd/system/
sudo systemctl enable --now signs2speech@$USER
journalctl -u signs2speech@$USER -f
```

---

## 10. Repo map and command cheat sheet

```
common/s2s/camera.py     open Pi camera / USB webcam / video file (all scripts use this)
common/s2s/landmarks.py  MediaPipe hand + pose -> FrameLandmarks (150 raw values/frame)
common/s2s/features.py   shoulder normalization + velocities -> 300 features/frame; resample; mirror
common/s2s/clips.py      .npz clip save/load + metadata (sign, signer, session, fps, source, width, height)
common/s2s/vocab.py      the sign list, control signs, spoken text
scripts/check_hw.py      hardware bring-up test (camera/landmarks/FPS, --speaker)
scripts/collect.py       record labeled clips
scripts/download_models.sh  fetch MediaPipe .task models
training/train.py        train + export TFLite + confusion matrix
pi/run.py                device runtime
pi/SETUP.md              short bring-up steps
pi/signs2speech@.service systemd boot-to-app template
docs/PROJECT_PLAN.md     original plan: architecture, BOM, week-by-week schedule, risks
docs/PROJECT_OBJECTIVES.md  objectives + measurable targets
docs/RESEARCH_AND_SETUP_LOG.md  this file
```

| Task | Command |
|---|---|
| Activate env (Pi) | `cd ~/Sign2Text && source .venv/bin/activate` |
| Update code on the Pi | `cd ~/Sign2Text && git pull` |
| Camera list | `rpicam-hello --list-cameras` |
| Camera + FPS test | `python scripts/check_hw.py [--headless] [--hands-only] [--camera 0]` |
| Speaker test | `python scripts/check_hw.py --speaker [--volume 2] [--audio-device ...]` |
| Record clips | `python scripts/collect.py --signer <name> --session <tag> [--camera pi]` |
| Train | `python training/train.py` |
| Run | `python pi/run.py [--headless]` |
| Temperature / throttling | `vcgencmd measure_temp` · `vcgencmd get_throttled` |
| GPIO diagram | `pinout` |
| Audio cards | `aplay -l` |
| Remote desktop | `rpi-connect on && rpi-connect signin` → connect.raspberrypi.com |

---

## 11. Schedule status and course deadlines

**Where we are (Oct 24, 2026, Fall Week 5):** the hardware just arrived, so practically we're at
plan Week 1, which is Pi bring-up. That's about 3–4 weeks behind the original plan.

| Plan milestone | Original date | Status |
|---|---|---|
| M1: hardware works, clips record end to end | Oct 4 | Not yet. This is the immediate target. |
| M2: 15 or more signs recognized live on the Pi, 80% or better on team signers | Oct 18 | Not yet |
| M3: end-to-end demo v0.5 (sign to speech, camera only) | Nov 1 | At risk; needs re-baselining |
| Fall deliverable (demo + interim report) | Dec 4 | Still reachable with a compressed plan |
| M4: architecture frozen | Jan 17 | — |
| M5: standalone device, no laptop | Jan 31 | — |
| M6: feature complete | Feb 14 | — |
| M7: working product | Feb 28 | — |
| Final demo and report | Mar 8–12 | — |

**SCU calendar:**
- Fall classes end Dec 4; finals Dec 7–11; Thanksgiving Nov 23–27.
- Winter quarter runs Jan 4 – Mar 12; finals Mar 15–19.

**Course deadlines** (from Sahana's announcement, noted Oct 19):
- **Introduction & Background:** due Oct 25.
- **Project Specification Table and Updated Functional Diagram:** due Nov 1. Both are graded
  more closely, and you can submit a week early for feedback.

**Since the glove was dropped, the functional diagram must be vision only.** It should probably
also include power/battery and enclosure blocks.

**Compressed path, a suggestion to agree with the advisor:**
- **This week:** Pi bring-up (§7) in parallel with laptop data collection (§8).
- **Next week:** first model on 10–15 signs, running live on the Pi.
- **After that:** add TTS for an end-to-end demo, then scale the vocabulary and signers before
  Thanksgiving.

The glove track is gone, which frees Karan to co-own data collection and the model.

---

## 12. Open items and next steps

1. **Buy the speaker and jumper wires** (§3).
2. **Pi bring-up** through §7.8. Record the FPS, detection rate and temperature.
3. **Start recording clips on both laptops** (§8).
4. **Re-baseline the Fall schedule** for the advisor accountability plan. The plan's week table
   still has the glove column and dates from before the hardware arrived.
5. **Update the pitch deck and project abstract.** They still mention the Portenta H7 and the
   glove/Nano; the project is now Pi 5 and vision only.
6. **Course deliverables:** Intro & Background (Oct 25); spec table and updated vision-only
   functional diagram (Nov 1).
7. **Choose dataset storage** (Git LFS vs. shared drive) and back up clips.
8. **Get the vocabulary reviewed by an ASL-fluent person.** Ask the department to connect you
   with SCU ASL students or the Deaf community.
9. **First training run on TF 2.20.** Confirm the TFLite export works (§6 "not tested").

---

## 13. Troubleshooting reference

| Symptom | Likely cause / fix |
|---|---|
| `rpicam-hello --list-cameras` lists nothing | Power off and reseat both cable ends with latches fully closed. Flip the narrow end at the Pi. Check the camera end matches the original cable's orientation. |
| `check_hw.py`: "no Pi camera found" | Same as above, or the venv wasn't created with `--system-site-packages` so picamera2 isn't visible. Recreate it. |
| `numpy.dtype size changed`, picamera2 import errors | The card has the older Bookworm image. Reflash with the current default Raspberry Pi OS. |
| `throttled=0x50005` (or other values with bits 0/2) | Under-voltage. Use the 27 W supply, not the power bank or a phone charger. |
| `throttled=0x80008` | Thermal throttling. Check the cooler's fan cable and that the pads touch the chips. |
| FPS under 10 | Confirm 640×480, try `--hands-only` and `--headless`, close other apps. Consider the AI HAT+ last. |
| No hands detected | Lighting from the front, film removed from the lens, stand about 1 m away, hands within the frame. |
| `collect.py` exits about aspect ratio | The webcam isn't delivering 4:3. Pass `--width 640 --height 480`, or try another camera. |
| `aplay -l` doesn't show the amp | Overlay line missing or misspelled in `/boot/firmware/config.txt`, or you didn't reboot. |
| Amp card present but silent | Check the wiring (pins 2, 6, 12, 35, 40) and the speaker terminals. Test with `--audio-device plughw:CARD=<name>,DEV=0`. |
| "Channels count non available" from aplay | You played to `hw:` directly. Use `plughw:` or the `asound.conf` default, which converts mono to stereo. |
| `run.py`: "No trained model yet" | Expected until you've trained and copied the `data/models/` files (§9). |
| `train.py`: no `tf.lite` | Install the pinned TF: `pip install "tensorflow>=2.20,<2.21"`. |
| pip installs both opencv packages and `cv2` acts strangely | `pip uninstall opencv-python opencv-contrib-python`, then `pip install -r requirements.txt` again. |
| `ssh s2s.local` fails | Use the Pi's IP address from the router or hotspot list. Confirm SSH was enabled in Imager. |

---

## 14. Reference data: model files, pinout, links

**MediaPipe model files** (in `data/models/`, gitignored, fetched by `scripts/download_models.sh`):

| File | URL | Size | SHA-256 |
|---|---|---|---|
| `hand_landmarker.task` | https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task | 7,819,105 B | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` |
| `pose_landmarker_lite.task` | https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task | 5,777,746 B | `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a` |

Verify a download with `sha256sum data/models/*.task`.

**Piper voice:** `en_US-lessac-medium`, downloaded with `python -m piper.download_voices`. It
creates two files, `en_US-lessac-medium.onnx` and `en_US-lessac-medium.onnx.json`, and the
`.json` holds the sample rate.

**MAX98357A ↔ Pi 5:** Vin to pin 2 (5 V), GND to pin 6, BCLK to pin 12 (GPIO 18), LRC to pin 35
(GPIO 19), DIN to pin 40 (GPIO 21). Overlay: `dtoverlay=max98357a`.

**Sources used in this research:**
- Adafruit MAX98357A, Raspberry Pi wiring: https://learn.adafruit.com/adafruit-max98357-i2s-class-d-mono-amp/raspberry-pi-wiring
- Adafruit MAX98357A, Raspberry Pi usage (overlay, asound.conf, mono note): https://learn.adafruit.com/adafruit-max98357-i2s-class-d-mono-amp/raspberry-pi-usage
- Raspberry Pi Connect documentation: https://www.raspberrypi.com/documentation/services/connect.html
- Raspberry Pi 5 Case (built-in fan and heatsink): https://www.raspberrypi.com/products/raspberry-pi-5-case/
- What's new in TensorFlow 2.20 (tf.lite deprecation): https://blog.tensorflow.org/2025/08/whats-new-in-tensorflow-2-20.html
- PyPI pages checked Oct 24, 2026 for versions and wheel platforms: https://pypi.org/project/mediapipe/ · https://pypi.org/project/ai-edge-litert/ · https://pypi.org/project/tflite-runtime/ · https://pypi.org/project/piper-tts/ · https://pypi.org/project/tensorflow/
- SunFounder MediaPipe setup (the outdated "no Python 3.13" advice): https://docs.sunfounder.com/projects/ai-lab-kit/en/latest/mediapipe/mp_0_setup.html
- Original plan: `docs/PROJECT_PLAN.md`. Reference repo: https://github.com/Muhib-Mehdi/ASL-Recognition-System

---

## 15. Earlier chat threads

- **"ASL to speech capstone project timeline"** (Sept–Oct 2026): architecture choice, plan, BOM,
  objectives, code skeleton, GitHub setup. https://claude.ai/chat/a03e5fc1-6652-4eda-a537-62efa1ed5b88
- **"Senior design project pitch presentation"** (Oct 2026): 7-slide pitch deck plus peer-review
  slides, weekly status report, functional diagram, glove dropped.
  https://claude.ai/chat/38689fce-d2eb-44c3-aacb-47288d86996e
- **This thread** (Oct 24, 2026): hardware purchased, software stack re-verified, bring-up code
  and guides, and this log.
