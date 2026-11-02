import html, os
OUT = os.path.dirname(os.path.abspath(__file__))
FONT = "Helvetica, Arial, sans-serif"
INK = "#1f2a44"; GREY = "#5b6478"; BLUE = "#2f5bd3"; FILL = "#eef3ff"; PALE = "#f6f7fa"; RED = "#a6192e"

def esc(s): return html.escape(s, quote=False)

class SVG:
    def __init__(s, w, h): s.w, s.h, s.p = w, h, []
    def add(s, x): s.p.append(x)
    def rect(s, x, y, w, h, fill="#fff", stroke=INK, sw=1.6, dash=None, rx=6):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        s.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
    def text(s, x, y, t, size=15, weight="normal", fill=INK, anchor="middle", italic=False):
        st = ' font-style="italic"' if italic else ""
        s.add(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{st}>{esc(t)}</text>')
    def block(s, x, y, w, h, title, lines=(), spec=None, dash=None, fill="#fff", stroke=INK):
        s.rect(x, y, w, h, fill=fill, stroke=stroke, dash=dash)
        n = 1 + len(lines) + (1 if spec else 0)
        lh = 19
        y0 = y + h/2 - (n-1)*lh/2 + 5
        s.text(x+w/2, y0, title, 16, "bold", INK if not dash else GREY)
        for i, l in enumerate(lines):
            s.text(x+w/2, y0+(i+1)*lh, l, 13.5, fill=GREY if dash else INK)
        if spec:
            s.text(x+w/2, y0+(len(lines)+1)*lh, spec, 13, "bold", BLUE if not dash else GREY)
    def arrow(s, pts, dash=None, color=INK, sw=1.8):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        path = "M " + " L ".join(f"{x} {y}" for x, y in pts)
        mk = "arrowG" if color != INK else "arrow"
        s.add(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{sw}"{d} marker-end="url(#{mk})"/>')
    def label(s, x, y, lines, size=13, fill=RED, anchor="middle"):
        for i, l in enumerate(lines):
            s.text(x, y + i*16, l, size, "normal", fill, anchor)
    def svg(s):
        defs = f'''<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{INK}"/></marker>
<marker id="arrowG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{GREY}"/></marker>
</defs>'''
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{s.w}" height="{s.h}" viewBox="0 0 {s.w} {s.h}"><rect width="100%" height="100%" fill="#fff"/>{defs}{"".join(s.p)}</svg>'

# ---------------- Level 1 ----------------
L = SVG(1960, 1010)
L.text(980, 40, "Level 1 Functional Block Diagram — Signs to Speech", 24, "bold")
BX0, BX1, BY0, BY1 = 280, 1700, 130, 830
L.rect(BX0, BY0, BX1 - BX0, BY1 - BY0, fill="#fbfcfe", stroke=INK, sw=2.2, rx=10)
L.text(BX0 + 20, BY0 + 28, "Signs to Speech device — Raspberry Pi 5 (all recognition on device, no cloud)", 15, "bold", INK, "start")

BW, BH = 200, 120
RA, RB, RC = 200, 420, 640
xs = {"cam": 320, "lmk": 645, "feat": 970, "clf": 1295}
L.block(xs["cam"], RA, BW, BH, "Image Capture", ["Camera Module 3 Wide", "120° view, autofocus"])
L.block(xs["lmk"], RA, BW, BH, "Landmark Extraction", ["MediaPipe hands + pose"], "≥ 10 fps (15.0 measured)", fill=FILL, stroke=BLUE)
L.block(xs["feat"], RA, BW, BH, "Feature Builder", ["shoulder-normalized", "points + velocity"], "40-frame window ≈ 2 s", fill=FILL, stroke=BLUE)
L.block(xs["clf"], RA, BW, BH, "Sign Classifier", ["1D-CNN + GRU on LiteRT"], "≥ 25 signs · ≥ 90 % acc.", fill=FILL, stroke=BLUE)
ym = RA + BH / 2
for a, b, lab in [("cam", "lmk", ["RGB frames", "640×480 @ ~15 fps"]),
                  ("lmk", "feat", ["150 values/frame", "(hands + pose)"]),
                  ("feat", "clf", ["40 × 300", "feature window"])]:
    x1, x2 = xs[a] + BW, xs[b]
    L.arrow([(x1, ym), (x2 - 2, ym)])
    L.label((x1 + x2) / 2, ym - 30, lab)

# row B: decision (under classifier) -> caption server (under feature builder)
L.block(xs["clf"], RB, BW, BH, "Decision & Sentence", ["confidence ≥ 0.8, debounce,", "idle class, SPEAK / CLEAR"], "≤ 1 s sign → caption", fill=FILL, stroke=BLUE)
L.arrow([(xs["clf"] + BW / 2, RA + BH), (xs["clf"] + BW / 2, RB - 2)])
L.label(xs["clf"] + BW / 2 + 14, (RA + BH + RB) / 2 - 4, ["31 class", "probabilities"], anchor="start")
L.block(xs["feat"], RB, BW, BH, "Caption Server", ["web page + live updates", "(HTTP / SSE)"], "captions ≥ 36 px", fill=FILL, stroke=BLUE)
L.arrow([(xs["clf"], RB + BH / 2), (xs["feat"] + BW + 2, RB + BH / 2)])
L.label((xs["feat"] + BW + xs["clf"]) / 2, RB + BH / 2 - 30, ["words &", "sentences"])

# display: external, below the boundary
DY = 880
L.block(xs["feat"], DY, BW, 100, "Display", ["laptop or phone browser"], "live captions", fill=PALE)
cx = xs["feat"] + BW / 2
L.arrow([(cx, RB + BH), (cx, DY - 2)])
L.label(cx + 14, 760, ["captions over Wi-Fi /", "Ethernet (local network)"], anchor="start")

# row C: phase-2 speech
L.block(xs["clf"], RC, BW, 100, "Text-to-Speech", ["Piper (offline)"], "phase 2", dash="7 5")
L.arrow([(xs["clf"] + BW / 2, RB + BH), (xs["clf"] + BW / 2, RC - 2)], dash="7 5", color=GREY)
L.label(xs["clf"] + BW / 2 + 14, (RB + BH + RC) / 2 + 4, ["text"], fill=GREY, anchor="start")
L.block(1740, RC, 200, 100, "Amplifier + Speaker", ["MAX98357A, ~3 W"], "phase 2", dash="7 5")
L.arrow([(xs["clf"] + BW, RC + 50), (1738, RC + 50)], dash="7 5", color=GREY)
L.label((xs["clf"] + BW + 1740) / 2, RC + 22, ["I2S audio", "22.05 kHz"], fill=GREY)

# external: signer
L.block(20, RA, 170, BH, "Signer", ["ASL user's hands,", "face, upper body"], fill=PALE)
L.arrow([(190, ym), (xs["cam"] - 2, ym)])
L.label(235, ym - 30, ["ASL signs", "(light)"])

# offline training, above the classifier
L.block(xs["clf"], 58, BW, 58, "Model Training (offline)", ["team laptop · TensorFlow"], dash="7 5")
tx = xs["clf"] + BW - 40
L.arrow([(tx, 116), (tx, RA - 2)], dash="7 5", color=GREY)
L.label(tx + 10, 160, ["model file", "(.tflite)"], fill=GREY, anchor="start")

# power supply + cooler
L.block(xs["cam"], RC, BW, 100, "Power Supply", ["Pi 5 power management"], "5 V / 3.3 V DC out")
L.block(xs["lmk"], RC, BW, 100, "Active Cooler", ["heatsink + fan"], "CPU ≤ 80 °C")
busY = 370
L.add(f'<path d="M {xs["cam"]+BW/2} {RC} L {xs["cam"]+BW/2} {busY} L {xs["feat"]+BW/2} {busY}" fill="none" stroke="{INK}" stroke-width="1.6"/>')
for k in ("cam", "lmk", "feat"):
    L.arrow([(xs[k] + BW / 2, busY), (xs[k] + BW / 2, RA + BH + 2)])
L.label(xs["cam"] + BW / 2 + 12, busY + 22, ["DC power to all subsystems"], anchor="start")
L.arrow([(xs["cam"] + BW, RC + 50), (xs["lmk"] - 2, RC + 50)])
L.label((xs["cam"] + BW + xs["lmk"]) / 2, RC + 40, ["5 V fan"])
L.block(20, RC, 170, 100, "Power Source", ["27 W USB-C supply", "or USB-C power bank"], "≥ 2 h on battery", fill=PALE)
L.arrow([(190, RC + 50), (xs["cam"] - 2, RC + 50)])
L.label(235, RC + 40, ["5 V / 5 A"])
hx = xs["lmk"] + BW / 2
L.arrow([(hx, RC + 100), (hx, 900)])
L.label(hx + 12, 895, ["heat out"], anchor="start")

# legend
LY = 985
L.add(f'<line x1="20" y1="{LY-5}" x2="60" y2="{LY-5}" stroke="{INK}" stroke-width="1.8"/>')
L.text(68, LY, "built: data / power path", 13, anchor="start")
L.add(f'<line x1="250" y1="{LY-5}" x2="290" y2="{LY-5}" stroke="{GREY}" stroke-width="1.8" stroke-dasharray="7 5"/>')
L.text(298, LY, "offline or phase 2", 13, anchor="start", fill=GREY)
L.rect(450, LY - 15, 28, 18, fill=FILL, stroke=BLUE, rx=3)
L.text(486, LY, "software on the Pi 5", 13, anchor="start")
L.text(1230, LY, "Red labels: the signal on each line and its quantity. Blue: the spec each block must meet.", 13, anchor="start", fill=RED)
open(os.path.join(OUT, "level1.svg"), "w").write(L.svg())

# ---------------- Level 0 ----------------
Z = SVG(1300, 470)
Z.text(650, 40, "Level 0 — Signs to Speech as a single block", 22, "bold")
Z.rect(470, 110, 360, 250, fill=FILL, stroke=BLUE, sw=2.2, rx=14)
Z.text(650, 225, "Signs to Speech", 26, "bold")
Z.text(650, 258, "edge ASL-to-caption device", 16, fill=GREY)
for y, lab in [(170, ["ASL signs (hand, face and", "body motion), 0.8–1.5 m"]), (300, ["Power: 5 V / 5 A USB-C", "(supply or power bank)"])]:
    Z.arrow([(130, y), (468, y)])
    Z.label(140, y - 30, lab, size=15, fill=INK, anchor="start")
for y, lab, dash in [(150, ["Live captions on laptop/phone", "≤ 1 s after each sign"], None),
                     (240, ["Speech (phase 2)"], "7 5"),
                     (320, ["Heat"], None)]:
    Z.arrow([(832, y), (1170, y)], dash=dash, color=GREY if dash else INK)
    Z.label(845, y - (30 if len(lab) > 1 else 12), lab, size=15, fill=GREY if dash else INK, anchor="start")
Z.arrow([(650, 440), (650, 362)])
Z.label(665, 430, ["Environment: indoor lighting"], size=15, fill=INK, anchor="start")
open(os.path.join(OUT, "level0.svg"), "w").write(Z.svg())
print("ok")
