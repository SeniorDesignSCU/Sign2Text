# Signs to Speech: Project Status

**As of Sunday, Oct 25, 2026, end of day.** This is a one-page handoff. To continue in a new
chat, share this file, or the repo link, first.

**Repo:** https://github.com/SeniorDesignSCU/Sign2Text (`main`, latest commit `5996bed`)
**Team:** Aditya Thakrar (ECE), Karan Velayutham (CSE) · SCU ECEN & CSEN cross-disciplinary
senior design, 2026–27
**Advisors:** Prof. Michael Schimpf, Fr. Dat Tran, S.J. · **Instructor:** Sahana

---

## What it is

An on-device ASL recognizer. A **Raspberry Pi 5 + Camera Module 3 Wide** tracks hands and upper
body with MediaPipe. A temporal model (1D-CNN + GRU, running on LiteRT) classifies **word-level
signs**, about 28 signs plus the CLEAR and SPEAK control signs. The result appears as **live
captions in a laptop browser** at `http://s2s.local:8000`; the laptop is only a screen, and all
processing runs on the Pi.

**Targets:**
- 90% or better accuracy on team signers, 80% or better on unseen signers
- 1 s or less from end of sign to output
- 10 FPS or more on the device
- 5 or more test users
- A standalone device

## Decisions in force

| Decision | Notes |
|---|---|
| Pi 5 + camera, not Portenta H7 | The Portenta can't run MediaPipe |
| **Vision only.** No glove. | Decided Oct 2026 |
| **Output = on-screen text in a laptop browser** for now | No display hardware will be bought |
| **Speech later** | Piper TTS + MAX98357A amp. The code exists behind `--speak`; it needs a speaker (4–8 Ω, ~3 W) and jumper wires |
| Raspberry Pi OS 64-bit (Trixie, Python 3.13) | MediaPipe 1.1.1 works on it |
| **Record and run at 640×480 (4:3)** | Training and Pi data must match; `collect.py` enforces it |
| TensorFlow 2.20.x for training on laptops | `tf.lite` is deprecated after 2.20 |

## Where things stand

**Done:**
- **Hardware bought:**
  - Raspberry Pi 5 16 GB, 27 W power supply, Active Cooler, Camera Module 3 Wide, Pi 5 camera
    cable, case, SanDisk 64 GB microSD
  - Logitech C270 webcam, power bank
  - MAX98357A amp (unused for now)
- **Code (all pushed):**
  - camera module (Pi camera, USB webcam, or video file)
  - landmark and feature pipeline
  - `collect.py` (data recording)
  - `train.py` (training and TFLite export)
  - `pi/run.py` (runtime, with the caption web page)
  - `scripts/check_hw.py` (hardware and caption checks)
- **Pi bring-up, Oct 25:**
  - Flashed with Imager 2.0.12; hostname `s2s`, user `adityathakrar`.
  - On Ethernet at **192.168.86.75**.
  - Camera `imx708_wide` detected.
  - **15.0 FPS** with hand and pose tracking, which passes the 10 FPS target.
  - 48.8 °C under load, `throttled=0x0`.
  - Live camera and landmarks view working in Chrome on the Mac.

**Not done or untested:**
- **No training clips recorded yet.** This is the critical path.
- **No model trained yet.** `train.py` hasn't been run on TF 2.20 yet.
- The caption demo (`check_hw.py --web --captions`) hasn't been viewed on real hardware yet.
- Whether campus Wi-Fi lets the Mac reach the Pi is unknown. If not, use the Pi's hotspot.
- The Pi's IP isn't reserved in the Google Home app yet.

## Open decision (was being asked when the thread ended)

**How to record training clips:**
- **(A) Mac webcam with `collect.py`.** Works today, but the camera differs from the Pi's
  wide-angle camera.
- **(B) Build a browser recording mode on the Pi (recommended).** A Record button and sign
  picker on the same web page, recording through the Pi camera. Training data would then match
  the deployed camera. `collect.py` can't run on the Pi over SSH because it needs a local
  window.

## Next steps, in order

1. Decide A or B above. If B, build the browser recording mode.
2. Record a first round: about 10–15 signs, 40 clips each per signer, plus lots of `NO_SIGN`.
   Camera level at chest height, about 1 m away.
3. Train on a laptop with `python training/train.py`. Copy the model to the Pi and run
   `python pi/run.py`. First live captions.
4. Scale to 25 or more signs and 3 or more extra signers. Get the vocabulary reviewed by an
   ASL-fluent person.
5. Re-baseline the Fall schedule for the advisor. Hardware arrived in Week 5, so the team is
   about 3–4 weeks behind the original plan.
6. **Course deadlines:**
   - Project Specification Table and Updated Functional Diagram due **Nov 1**. The diagram
     must be vision only, with the laptop as the display and speech as a later output.
   - Update the pitch deck and abstract, which still mention the Portenta and glove.
7. Later: buy the speaker and wires, follow `pi/SETUP.md` step 9, and add `--speak`.

## Where everything is

| File | Use it for |
|---|---|
| `pi/QUICK_REFERENCE.md` | Day-to-day commands: start, SSH, checks, run, copy files, shut down, troubleshooting, starting over |
| `pi/SETUP.md` | First-time setup, step by step with checks |
| `docs/RESEARCH_AND_SETUP_LOG.md` | Everything researched: decisions, verified versions, bugs found, full walkthrough, bring-up log (§16), sources |
| `docs/PROJECT_PLAN.md` | Original plan: architecture, BOM, week-by-week schedule, risks (has an Oct 24 status note at top) |
| `docs/PROJECT_OBJECTIVES.md` | Objectives and measurable targets |
| `README.md` | Overview, layout, laptop quick start |

**Resume on the Pi** (cable in a router `<··>` port, then power):
```
ssh adityathakrar@s2s.local
```
```
cd ~/Sign2Text && source .venv/bin/activate && git pull
```
