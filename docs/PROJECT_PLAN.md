# Signs to Speech — Senior Design Plan
**Edge-processed ASL word/phrase recognition with speech output**
Team: Aditya, Karan Velayutham · Santa Clara University ECEN · Fall 2026 – Winter 2027
Prepared for advisor meeting, Sept 16, 2026 · Project start Sept 22, 2026 · Working product target: Feb 28, 2027

---

## 1. Recommended architecture

**Primary sensor: camera + landmark extraction on a Raspberry Pi 5. Secondary sensor: instrumented glove on an Arduino Nano 33 BLE Sense Rev2. All inference on-device; no cloud.**

Why this and not the Portenta H7 + Vision Shield in the original proposal:

- Word/phrase signs are *dynamic*, frequently *two-handed*, and depend on *where the hand is relative to the face and torso* (e.g., THANK-YOU starts at the chin; SORRY circles on the chest). A single glove captures handshape and wrist motion but not location or the second hand. A camera with hand + pose landmarks captures all three. The camera has to be the primary sensor for a word-level vocabulary.
- The Portenta H7 is a microcontroller. It cannot run MediaPipe; its Vision Shield gives a 324×324 grayscale image that would be classified as raw pixels at ~96×96. That approach discards everything that makes the reference repo work (landmark extraction) and is far less robust to lighting and background. The Pi 5 runs Google's own MediaPipe landmarker examples for Raspberry Pi and is still edge processing — small, battery-capable, offline.
- The glove stays in the design because it is the team's genuine differentiator: it disambiguates handshape-confusable signs, gives a lighting-independent fallback, and lets us demonstrate sensor fusion. It is scheduled as a secondary track so it cannot block the critical path.

### Relationship to the reference repo (Muhib-Mehdi/ASL-Recognition-System)
The repo does static A–Z fingerspelling on a laptop: MediaPipe → 21 landmarks → normalize → small dense network → letter. Its own roadmap lists dynamic gestures, word-level recognition, text-to-speech, and edge deployment as *unfinished*. Our project delivers exactly those four things, so the contribution is clear and nothing is copied:

| Reference repo | Our system |
|---|---|
| Static letters, single frame | Dynamic word/phrase signs, sliding window of frames |
| Hand landmarks only | Hand + upper-body pose landmarks (location, two hands) |
| Dense MLP on 42 features | Temporal model (1D-CNN / GRU) on landmark sequences, TFLite |
| Runs on a laptop | Runs standalone on Raspberry Pi 5, boots into the app |
| Text on screen | Offline text-to-speech through a speaker |
| Vision only | Vision + IMU/flex glove, late fusion |

We keep the *idea* of wrist-relative, scale-normalized landmarks and write our own pipeline. Fingerspelling A–Z can be added late as a secondary mode (for names), which is the one place the repo's approach transfers directly.

### Pipeline

```
Camera (Pi Camera Module 3 Wide)
   → MediaPipe Hand Landmarker + Pose Landmarker (Pi 5, ~10–20 FPS on CPU)
   → Normalize: translate to shoulder midpoint, scale by shoulder width, add velocities
   → Sliding window (~30–45 frames ≈ 1.5–2 s) → temporal model (TFLite)
   → Sign probability → confidence threshold + debounce + "no-sign" class
   → Word buffer → speak on pause (Piper TTS, offline) → speaker + on-screen text

Glove (Nano 33 BLE Sense Rev2): 5 flex sensors + 9-axis IMU @ 50 Hz
   → on-device handshape classifier (Edge Impulse, TFLite Micro)
   → BLE → Pi → late fusion (weighted probabilities) for handshape-confusable signs
```

### Public data to bootstrap and benchmark
- **Google Isolated Sign Language Recognition (Kaggle, 2023)** — MediaPipe landmark sequences for 250 ASL signs. Directly comparable to our feature representation; useful for pretraining and for reading winning architectures. Check license before use in the report.
- **WLASL** — word-level ASL video dataset (~2,000 glosses). Useful for choosing vocabulary and for extra training clips run through our landmark extractor.
- **Lifeprint / ASL University** — reference for correct sign execution when recording.

---

## 2. Scope

**Vocabulary target:** 25–30 conversational signs. Proposed starter list (adjust with an ASL-fluent reviewer in Week 0):
HELLO, THANK-YOU, PLEASE, SORRY, YES, NO, HELP, NAME, WHAT, WHERE, WANT, NEED, EAT, DRINK, WATER, BATHROOM, MORE, STOP, GOOD, BAD, I/ME, YOU, LOVE, FRIEND, HOME, WORK, LEARN, TIME, plus control signs CLEAR and SPEAK.

| Tier | Deliverable |
|---|---|
| **Must (MVP)** | Real-time recognition of ≥25 signs on the Pi; spoken output through a speaker; standalone device (no laptop); ≥90% top-1 accuracy on team signers, ≥80% on signers not in the training set; ≤1 s from end of sign to speech |
| **Should** | Glove fusion improving accuracy on ≥3 confusable pairs; static A–Z fingerspelling mode; on-screen text; wiring diagram, setup scripts, README |
| **Stretch** | Sentence buffering with simple grammar smoothing; AI HAT+ acceleration; second glove; user-corrected retraining |

Explicit non-goals (state these in the report): full ASL grammar/translation, facial-expression grammar, continuous signing without pauses.

---

## 3. Hardware bill of materials

Self-funded by the team. Core build ≈ $430; with optional items ≈ $650. Prices are approximate and Raspberry Pi board prices have moved recently — check the linked page at order time. Raspberry Pi official pages route "Buy now" to approved US resellers (PiShop.us, Adafruit, CanaKit, Vilros); Arduino and SparkFun/Adafruit sell direct.

### Compute + vision
| Item | Qty | Est. | Official link | Notes |
|---|---|---|---|---|
| Raspberry Pi 5, 8 GB | 1 | $80 | [raspberrypi.com/products/raspberry-pi-5](https://www.raspberrypi.com/products/raspberry-pi-5/) | 8 GB — MediaPipe + TTS + camera stack is memory-hungry |
| Raspberry Pi 27 W USB-C Power Supply | 1 | $12 | [raspberrypi.com/products/27w-power-supply](https://www.raspberrypi.com/products/27w-power-supply/) | Required for Pi 5 under sustained load |
| Raspberry Pi Active Cooler | 1 | $5 | [raspberrypi.com/products/active-cooler](https://www.raspberrypi.com/products/active-cooler/) | Sustained inference throttles without it |
| Raspberry Pi Camera Module 3 **Wide** | 1 | $35 | [raspberrypi.com/products/camera-module-3](https://www.raspberrypi.com/products/camera-module-3/) (choose Wide variant) | 120° FOV captures torso + both hands at arm's length; autofocus |
| Pi 5 camera cable, 22-pin to 15-pin, 200 mm | 1 | $3 | [adafruit.com — Raspberry Pi category](https://www.adafruit.com/category/360) | Pi 5 uses the smaller MIPI connector; some camera bundles now include it — check before ordering |
| Raspberry Pi Case for Raspberry Pi 5 | 1 | $10 | [raspberrypi.com/products/raspberry-pi-5-case](https://www.raspberrypi.com/products/raspberry-pi-5-case/) | Fan-compatible; works with the Active Cooler |
| microSD 64 GB, A2 class | 2 | $24 | Any reputable brand (SanDisk / Samsung) | One spare / one for a clean image |
| Audio: Adafruit I2S 3 W Class D Amp (MAX98357A) + 4 Ω 3 W speaker | 1 | $15 | [adafruit.com/product/3006](https://www.adafruit.com/product/3006) | Pi 5 has no 3.5 mm jack. A powered USB speaker is a simpler alternative |
| USB-PD power bank, 27 W-capable | 1 | $35 | Anker / UGREEN or similar | Portable demos; must negotiate 5 V 5 A |

### Glove
| Item | Qty | Est. | Official link | Notes |
|---|---|---|---|---|
| Arduino Nano 33 BLE Sense Rev2 | 1 | $39 | [store.arduino.cc/products/nano-33-ble-sense-rev2](https://store.arduino.cc/products/nano-33-ble-sense-rev2) | IMU (BMI270 + BMM150), BLE, TFLite Micro capable; "with headers" variant saves soldering |
| Flex Sensor 2.2" (Spectra Symbol, SparkFun SEN-10264) | 6 | $60 | [sparkfun.com/flex-sensor-2-2.html](https://www.sparkfun.com/flex-sensor-2-2.html) | 5 fingers + 1 spare; do not bend at the base |
| 10 kΩ resistors, jumper wire, perfboard, header pins, heat shrink | — | $12 | Any electronics supplier | Voltage divider per sensor into A0–A4 |
| Thin, snug gloves (cycling/lycra) | 2 | $15 | Any | One to build, one to rebuild |
| Adafruit PowerBoost 1000C (LiPo charger + 5 V boost) | 1 | $20 | [adafruit.com/product/2465](https://www.adafruit.com/product/2465) | Feeds 5 V into the Nano's VIN; charges over micro-USB |
| 1S LiPo, 1000 mAh, JST-PH | 1 | $10 | [adafruit.com — batteries category](https://www.adafruit.com/category/918) | Pick a 1000–1200 mAh cell with a JST-PH plug to match the PowerBoost |
| Velcro, thread, hot glue, cable ties | — | $10 | Any | |

### Development / data collection
| Item | Qty | Est. | Notes |
|---|---|---|---|
| USB webcam, 720p+ | 1 | $25 | Lets both teammates collect data on laptops without the Pi |
| USB-C cables, USB-A to micro-USB (Nano) | — | $10 | |

### Optional (buy only if needed)
| Item | Est. | Official link | Trigger |
|---|---|---|---|
| Raspberry Pi AI HAT+ (Hailo, 13 or 26 TOPS) | $70–110 | [raspberrypi.com/products/ai-hat](https://www.raspberrypi.com/products/ai-hat/) | Only if Pi 5 CPU FPS is below ~10 after optimization (Week 4 checkpoint). Fits with the Active Cooler in place |
| Small HDMI/touch display for on-device text | $70–90 | Raspberry Pi Touch Display 2 via approved resellers | If a built-in text display is wanted for demos; otherwise a phone/laptop browser page |
| Second Nano 33 BLE Sense Rev2 + sensors | $100 | Links above | Two-glove stretch goal |

Dropping the Portenta Machine Vision Bundle saves $151. If one is already owned, keep it as a backup platform for a mini side-experiment (image-classification baseline) in the report.

## 4. Software stack

| Layer | Choice | Reason |
|---|---|---|
| Pi OS | Raspberry Pi OS 64-bit (Bookworm), Python 3.11 | Required for MediaPipe aarch64 wheels and picamera2 |
| Landmarks | MediaPipe Tasks: Hand Landmarker + Pose Landmarker | Google maintains a Raspberry Pi example for these |
| Camera | picamera2 (CSI) / OpenCV (USB) | |
| Training | TensorFlow/Keras on laptops or Colab; export TFLite | Same toolchain as reference repo, easy to explain |
| On-Pi inference | tflite-runtime | |
| TTS | Piper (offline neural TTS) | Runs on Pi 5 in real time, no network |
| Glove firmware | Arduino (C/C++) via PlatformIO in VS Code | Answers the "VS Code" question from the Sept 8 notes |
| Glove ML | Edge Impulse | Officially supports Nano 33 BLE Sense; exports an Arduino library; free developer tier. Preferred over SensiML for community size and lower friction |
| BLE bridge | `bleak` (Python) on Pi | |
| Repo / tracking | GitHub, GitHub Projects board, one issue per weekly task | Advisor can see progress directly |

Data format: one file per clip (`signer_id, session, sign, fps, landmarks[frame][point][x,y,z], glove[frame][...]`) as Parquet/NumPy; metadata CSV. This is the single most valuable artifact of the project — treat it like code (versioned, backed up).

---

## 5. Roles (proposed, swappable)

- **Aditya** — vision pipeline, temporal model, Pi integration, TTS, systemd/boot.
- **Karan** — glove hardware and firmware, Edge Impulse model, BLE link, data-collection tooling.
- **Both** — data collection, user testing, report, presentations. Weekly 30-min sync; task board updated before each advisor meeting.

---

## 6. Week-by-week schedule

Dates from the SCU 2026-27 undergraduate calendar: Fall classes Sept 21 – Dec 4, finals Dec 7–11, Thanksgiving Nov 23–27. Winter classes Jan 4 – Mar 12, finals Mar 15–19, MLK Jan 18, Presidents' Day Feb 15.

### Fall Quarter 2026 — build the vision MVP, prototype the glove

| Wk | Dates | Vision / integration (Aditya) | Glove (Karan) | Check-in deliverable |
|---|---|---|---|---|
| 0 | Sep 14–20 | Advisor meeting Sep 16; finalize architecture; **order all parts**; create repo + project board; pick 25-sign vocabulary; read Kaggle ISLR write-ups | Order parts; set up PlatformIO + Edge Impulse; study flex-sensor wiring | Approved plan, parts ordered |
| 1 | Sep 21–27 | Pi bring-up: OS, camera, MediaPipe hand + pose landmarks live at ≥10 FPS; laptop data-collection script v0 (webcam → landmarks → clip file) | Nano bring-up; read IMU + 5 analog channels at 50 Hz over serial; plot in Serial Plotter | Live landmark video on Pi |
| 2 | Sep 28–Oct 4 | Data-collection tool v1: countdown, labeled clips, auto-trim, signer/session/lighting metadata; landmark normalization + velocity features | Solder flex sensors into glove v0; calibration routine (min/max per finger); BLE stream to laptop | **M1 (Oct 4): hardware works; can record labeled clips end to end** |
| 3 | Oct 5–11 | Dataset round 1: 25 signs × 2 signers × ~40 reps; first temporal model (1D-CNN or GRU); confusion matrix | Edge Impulse project; collect handshape data for ~10 handshapes; first classifier | First accuracy numbers |
| 4 | Oct 12–18 (Oct 12 holiday) | Iterate: augmentation (mirror, time-warp, jitter), window size, class balance; export TFLite; run on Pi; measure latency and FPS. **AI HAT+ decision point** | Improve glove dataset; try IMU-only vs flex-only vs both | **M2 (Oct 18): ≥15 signs recognized on Pi in real time, ≥80% on team signers** |
| 5 | Oct 19–25 | Real-time loop on Pi: sliding window, "no-sign" idle class, confidence threshold, debounce, on-screen text buffer | On-device inference on Nano (Edge Impulse Arduino library); BLE broadcast of class + confidence | Stable real-time predictions |
| 6 | Oct 26–Nov 1 | Piper TTS on Pi; speaker; word buffer → speak on pause; end-to-end latency budget | Define fusion message format and timing sync with Pi | **M3 (Nov 1): end-to-end demo v0.5 — sign → speech, camera only** |
| 7 | Nov 2–8 | Vocabulary to full 25–30; dataset round 2 with ≥3 additional signers (recruit ASL club/class students); retrain | Assist data collection; glove v1 rebuild with strain relief | Cross-signer accuracy number |
| 8 | Nov 9–15 | Fusion v1 on Pi: late fusion for confusable pairs; measure gain vs camera-only | Glove reliability testing; battery runtime | Fusion ablation table |
| 9 | Nov 16–22 | Robustness: lighting, background, distance, left-handed signers; fix top-5 failures; static A–Z fingerspelling mode (secondary) | Same; write glove build doc | Interim report draft |
| — | Nov 23–27 | Thanksgiving — no scheduled work | | |
| 10 | Nov 30–Dec 4 | Fall demo to advisor; interim report; freeze fall codebase; write winter backlog | | **Fall deliverable submitted** |
| — | Dec 7–11 | Finals — no project work | | |
| — | Dec 12–Jan 3 | Winter break — optional only: enclosure/glove v2 CAD, extra signer data | | |

### Winter Quarter 2027 — standalone device, testing, polish

| Wk | Dates | Vision / integration (Aditya) | Glove (Karan) | Check-in deliverable |
|---|---|---|---|---|
| 11 | Jan 4–10 | Regroup; re-run full test suite; confirm stretch list | Glove v2 (battery, strain relief, connectors) | Winter kickoff notes |
| 12 | Jan 11–17 | Model v2: both-hands + pose features, GRU/transformer variant; target ≥85% cross-signer | Glove v2 data recollection | **M4 (Jan 17): architecture and feature list frozen** |
| 13 | Jan 18–24 (Jan 18 holiday) | Sentence mode: buffering, repeat suppression, CLEAR/SPEAK control signs | BLE reconnection/robustness | Sentence demo |
| 14 | Jan 25–31 | Portable form factor: battery, mounted camera, speaker, boot-to-app (systemd), auto-start, status LED | Enclosure for glove electronics | **M5 (Jan 31): standalone device runs with no laptop** |
| 15 | Feb 1–7 | User test round 1: ≥5 testers incl. ≥1 fluent signer; scripted protocol (each sign ×10); log accuracy, latency, failure clips | Same | Test report v1 |
| 16 | Feb 8–14 | Retrain with new data; fix UX issues | Final fusion tuning **or descope decision** | **M6 (Feb 14): feature complete** |
| 17 | Feb 15–21 (Feb 15 holiday) | User test round 2; final metrics; start final report + poster | Same | Final metrics table |
| 18 | Feb 22–28 | Code freeze; README, setup scripts, wiring diagram; demo video | Same | **M7 (Feb 28): working product complete** |
| 19 | Mar 1–7 | Final report draft; presentation rehearsal; slip buffer | | Report draft to advisor |
| 20 | Mar 8–12 | Final demo/presentation; submit report; hand off repo | | **Final deliverable** |
| — | Mar 15–19 | Finals | | |

Weeks 19–20 are deliberately buffer. If the team is ahead (the advisor said early finish is fine), M7 can move up and Weeks 17–20 become polish.

---

## 7. Risks and mitigations

| Risk | Likelihood | Mitigation | Decision date |
|---|---|---|---|
| Pi 5 FPS too low for 30-frame windows | Medium | 640×480 input, hand-only landmarker for some signs, multithreading; AI HAT+ as fallback | Oct 18 |
| Poor cross-signer accuracy | High | Recruit ≥3 extra signers by Week 7; shoulder-normalized features; augmentation; pretrain on Kaggle ISLR landmarks | Nov 8 |
| Glove hardware unreliable (flex sensor fatigue, BLE drops) | Medium | Glove is a secondary track; spare sensors; hard descope date | Feb 14 |
| Two-handed signs / occlusion | Medium | Choose vocabulary in Week 0 with occlusion in mind; include "no-sign" class; pose landmarks give second-hand location | Sep 20 |
| Part delays | Low | Order everything Week 0; laptop + webcam path lets ML work start without the Pi | Sep 20 |
| Signs recorded incorrectly | Medium | Verify vocabulary with an ASL-fluent reviewer; use Lifeprint as reference | Oct 4 |
| Scope creep | High | Feature freeze Jan 17; stretch items only after M6 | Jan 17 |

---

## 8. Success metrics (report these at each milestone)

- Top-1 accuracy on held-out clips: team signers / unseen signers
- Real-time accuracy in scripted user tests (sign ×10 per tester)
- Latency: end of sign → speech start (ms)
- Throughput: landmark FPS on Pi
- Fusion ablation: camera-only vs camera + glove on confusable pairs
- Standalone runtime on battery (minutes)

---

## 9. Questions for the advisor on Sept 16

1. Approval to change compute platform from Portenta H7 + Vision Shield to Raspberry Pi 5 + Camera Module 3 (rationale in §1). Does the submitted abstract need to be amended?
2. Edge Impulse instead of SensiML for the glove model — any course requirement for SensiML?
3. Fixed course deliverable dates (design review, interim report, final presentation) so the schedule can be anchored to them.
4. Can the department connect us with SCU ASL students or Deaf community members for vocabulary review and testing?
5. Is there lab space and bench equipment (soldering station, oscilloscope) we can use for the glove build?
