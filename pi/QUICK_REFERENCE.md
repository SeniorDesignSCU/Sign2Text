# Pi Quick Reference

Copy-paste commands for working with the Signs to Speech Pi. These assume the **Pi is on
Ethernet**, plugged into the home Google Wi-Fi router. A Wi-Fi-only version is at the end.

- First-time setup with explanations: [SETUP.md](SETUP.md)
- Everything researched and decided so far: [../docs/RESEARCH_AND_SETUP_LOG.md](../docs/RESEARCH_AND_SETUP_LOG.md)

---

## Our setup

| | |
|---|---|
| Hostname | `s2s` → `s2s.local` |
| Username | `adityathakrar` (the password is not stored here) |
| IP on home network | `192.168.86.75` (Google Wi-Fi, `192.168.86.x`). Reserve it so it stays fixed; see the end of this file. |
| Project folder on Pi | `~/Sign2Text` |
| Caption / camera page | `http://s2s.local:8000`, or `http://192.168.86.75:8000` |
| Laptop to use | **Personal MacBook.** The Cisco work laptop's security software blocks finding the Pi. |
| Baseline (Oct 25, 2026) | 15.0 FPS hand and pose tracking, 48.8 °C under load, `throttled=0x0` |

---

## 1. Start the Pi

1. Plug the Ethernet cable into a router port marked **`<··>`** (LAN). **Don't use the globe
   port**; that one is for the internet line.
2. **Then** plug in the Pi's 27 W power supply.
3. Wait **1–2 minutes**. The green light settles to solid when it's ready.

Check from the Mac that it's up (Ctrl+C to stop):
```
ping s2s.local
```

## 2. Connect from the Mac

```
ssh adityathakrar@s2s.local
```
If that hangs or says "Unknown host", use the IP:
```
ssh adityathakrar@192.168.86.75
```

## 3. Every session: enter the project

Run this after every SSH login, before any project command:
```
cd ~/Sign2Text && source .venv/bin/activate
```
Your prompt then starts with `(.venv)`.

## 4. Get the latest code

When code has been pushed to GitHub:
```
git pull
```
If `requirements-pi.txt` changed in that pull, also run:
```
pip install -r requirements-pi.txt
```

## 5. Check the hardware

**Camera, landmarks and FPS, viewed on the Mac:**
```
python scripts/check_hw.py --web
```
Open `http://s2s.local:8000` on the Mac and click **Show camera**. Press Ctrl+C in SSH to stop
and print the result.

**Same test with no browser** (prints FPS for 20 seconds):
```
python scripts/check_hw.py
```

**Caption page with demo words** (no model needed):
```
python scripts/check_hw.py --web --captions
```

**FPS without pose tracking**, for comparison:
```
python scripts/check_hw.py --hands-only
```

**Using the Logitech C270** (plugged into a blue USB port) instead of the Pi camera:
```
python scripts/check_hw.py --web --camera 0
```

## 6. Quick health checks

**Is the camera detected?** It should list `imx708_wide`.
```
rpicam-hello --list-cameras
```

**Power problems?** It should say `throttled=0x0`.
```
vcgencmd get_throttled
```

**Temperature:** about 40–55 °C when idle; under about 75 °C when busy.
```
vcgencmd measure_temp
```

## 7. Run the recognizer (once a trained model is on the Pi)

```
python pi/run.py
```
Captions appear at `http://s2s.local:8000`, and recognized words also print in SSH.

**With the C270:**
```
python pi/run.py --camera 0
```

**With speech,** once the speaker is wired:
```
python pi/run.py --speak
```

## 8. Copy files between the Mac and the Pi

Run these **on the Mac**, in a new Terminal window, not inside SSH.

**Send a trained model to the Pi** (from the Mac's Sign2Text folder):
```
scp data/models/sign_classifier.tflite data/models/labels.txt data/models/model_config.json adityathakrar@s2s.local:~/Sign2Text/data/models/
```

**Copy recorded clips from the Pi to the Mac:**
```
scp -r adityathakrar@s2s.local:~/Sign2Text/data/clips ./data/
```

## 9. Shut down or restart

Always shut down before unplugging, so the SD card doesn't get corrupted.

**Shut down.** Wait until the green light goes off, then unplug the power.
```
sudo shutdown -h now
```

**Restart.** Wait about 1 minute, then SSH in again.
```
sudo reboot
```

**Update the OS** (about once a month):
```
sudo apt update && sudo apt full-upgrade -y
```

---

## If something goes wrong

| Problem | Fix |
|---|---|
| `ping` / `ssh s2s.local` says "Unknown host" or hangs | Wait 2 minutes after power-on. Use `192.168.86.75` instead. Check the Google Home app's device list for **s2s** (or `s2s-2`) and use the IP it shows. Make sure the cable is in a **`<··>`** port. Use the personal Mac with no VPN. |
| "REMOTE HOST IDENTIFICATION HAS CHANGED" when connecting | Normal after reflashing the card. On the Mac, run `ssh-keygen -R s2s.local` and `ssh-keygen -R 192.168.86.75`, then connect again. |
| Browser page won't load | Is `check_hw.py --web` or `run.py` still running in SSH? Try `http://192.168.86.75:8000`. |
| "Address already in use" | A previous run is still going. Stop it with Ctrl+C in its window, or run `pkill -f check_hw.py; pkill -f pi/run.py`, then try again. |
| `ModuleNotFoundError` (mediapipe, cv2 …) | You forgot step 3. Run `cd ~/Sign2Text && source .venv/bin/activate`. |
| Camera not listed | Shut down, unplug, and reseat the ribbon at both ends. Contacts face the camera board at the camera end, and face away from the flap at the Pi end. |
| `throttled` isn't `0x0` | Use the official 27 W supply, not a phone charger or power bank, and check that the cooler's fan cable is plugged in. |
| FPS under 10 | Close the browser's camera view; streaming costs a few FPS (about 15 without it, about 11 with it on Oct 25). Try `--hands-only` to compare. |

---

## Starting over (if the SD card is ever reflashed)

1. **Raspberry Pi Imager on the personal Mac**, with the card in the USB-C reader:
   - **Device:** Raspberry Pi 5.
   - **OS:** Raspberry Pi OS (64-bit).
   - **Storage:** the roughly 64 GB card.
   - **Hostname:** `s2s`.
   - **Localisation:** US, `America/Los_Angeles`, keyboard `us`.
   - **User:** `adityathakrar` plus a password.
   - **Wi-Fi:** home network.
   - **Remote access:** **switch Enable SSH on** (it's off by default) and choose password
     authentication.
   - **Before clicking Write,** the summary must list **SSH enabled**.
   - If macOS says the disk isn't readable after writing, click Ignore, never Initialize.
2. Do **section 1 (Start the Pi)**, then on the Mac run `ssh-keygen -R s2s.local`, then
   **section 2 (Connect)**.
3. **Update the system and install system packages,** one at a time:
   ```
   sudo apt update && sudo apt full-upgrade -y
   ```
   ```
   sudo apt install -y git python3-venv python3-picamera2 alsa-utils libportaudio2
   ```
   ```
   sudo reboot
   ```
4. **SSH in again and install the project,** one at a time:
   ```
   git clone https://github.com/SeniorDesignSCU/Sign2Text.git ~/Sign2Text
   ```
   ```
   cd ~/Sign2Text
   ```
   ```
   python3 -m venv --system-site-packages .venv
   ```
   ```
   source .venv/bin/activate
   ```
   ```
   pip install -r requirements-pi.txt
   ```
   ```
   bash scripts/download_models.sh
   ```
5. **Verify.** It should print `ok 1.1.1` or a newer version.
   ```
   python -c "import mediapipe, picamera2; from ai_edge_litert.interpreter import Interpreter; print('ok', mediapipe.__version__)"
   ```
   A red `ERROR: ... types-seaborn requires pandas-stubs` during `pip install` is harmless.
6. **Hardware check:** section 5.

---

## Without Ethernet (Wi-Fi only)

- **The Pi joins the Wi-Fi saved on the card.** To add or switch networks, connect once over
  Ethernet, then run:
  ```
  sudo nmtui
  ```
  Choose **Activate a connection**, pick the network, enter its password, and quit.
- **On campus,** SCU Wi-Fi may block the Mac from reaching the Pi. Use a phone hotspot (both
  devices on it), or have the Pi make its own network:
  ```
  sudo nmcli device wifi hotspot ifname wlan0 ssid S2S-Demo password "pick-a-password"
  ```
  Join `S2S-Demo` from the Mac and open `http://10.42.0.1:8000`. Turn it off with:
  ```
  sudo nmcli connection down Hotspot
  ```

## Keep the IP fixed (recommended)

In the **Google Home app**, go to Wi-Fi, then settings, then **Advanced networking**, then
**DHCP IP reservations**. Add the device **s2s** and reserve `192.168.86.75`. Menu names may
differ slightly by app version. After that, the IP in this file always works, even when
`s2s.local` doesn't.
