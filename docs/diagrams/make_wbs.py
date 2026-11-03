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


BLUE2, GREEN2, GREY2 = "#2f5bd3", "#1b8a5a", "#5b6478"
FILLS = {"Aditya": ("#e8eefc", BLUE2), "Karan": ("#e4f4ec", GREEN2), "Both": ("#f0f1f4", GREY2)}
P = [
 ("1.0", "Project management", "Both", ["1.1 Weekly reports (WARs)", "1.2 Advisor meetings", "1.3 Re-baseline schedule", "1.4 Budget tracking", "1.5 Hazard + budget forms", "1.6 Instructor meeting"]),
 ("2.0", "Requirements & design", "Both", ["2.1 Problem statement", "2.2 Architecture decision", "2.3 Specifications table", "2.4 Block diagrams", "2.5 Final vocabulary", "2.6 ASL-fluent review"]),
 ("3.0", "Hardware", "Aditya", ["3.1 Procure hardware", "3.2 Assemble", "3.3 OS + bring-up", "3.4 Camera mount", "3.5 Speaker + amplifier", "3.6 Portable build", "3.7 Battery runtime test"]),
 ("4.0", "Device software", "Aditya", ["4.1 Capture + landmarks", "4.2 Caption web page", "4.3 Browser recording mode", "4.4 Recognizer integration", "4.5 Sentence mode", "4.6 Latency + idle tuning", "4.7 Speech output", "4.8 Boot-to-app"]),
 ("5.0", "Dataset", "Karan", ["5.1 Recording protocol", "5.2 Storage + backup", "5.3 Round 1 recording", "5.4 Recruit 3+ signers", "5.5 Round 2 recording", "5.6 Data quality checks", "5.7 Round 3 recording"]),
 ("6.0", "Model", "Aditya", ["6.1 Verify pipeline", "6.2 Model v1 + deploy", "6.3 Retrain (round 2)", "6.4 Architecture freeze", "6.5 Optimize for Pi", "6.6 Final retrain"]),
 ("7.0", "Testing & validation", "Karan", ["7.1 Test plan", "7.2 Robustness tests", "7.3 Thermal test", "7.4 Accuracy tests", "7.5 Latency + idle tests", "7.6 User testing 1", "7.7 User testing 2", "7.8 Results analysis"]),
 ("8.0", "Thesis", "Both", ["8.1 Intro + background", "8.2 Specs + diagrams", "8.3 WBS + schedule", "8.4 Interim report", "8.5 Hardware chapter", "8.6 Software + model ch.", "8.7 Dataset chapter", "8.8 Testing + results ch.", "8.9 Ethics + conclusions", "8.10 Full draft + revise", "8.11 Final submission"]),
 ("9.0", "Presentations & demos", "Both", ["9.1 Pitch deck", "9.2 Fall demo", "9.3 Demo video", "9.4 Winter final demo", "9.5 Poster", "9.6 Conference slides", "9.7 Conference talk"]),
]
BW, GAP, X0 = 196, 14, 140
W = X0 + len(P) * BW + (len(P) - 1) * GAP + 30
LH = 20
H = 300 + max(len(p[3]) for p in P) * LH + 90
S = SVG(W, H)
S.text(W / 2, 38, "Signs to Speech — Work Breakdown Structure", 24, "bold")
# level labels
for y, t in [(110, "Project"), (215, "Work packages"), (310, "Activities")]:
    S.text(20, y, t, 14, "bold", GREY, "start")
# root
rw = 420; rx = (X0 + W - 30) / 2 - rw / 2
S.rect(rx, 80, rw, 60, fill=INK, stroke=INK, rx=8)
S.text(rx + rw / 2, 108, "Signs to Speech", 19, "bold", "#ffffff")
S.text(rx + rw / 2, 129, "Senior Design 2026–27 · Aditya Thakrar, Karan Velayutham", 12.5, fill="#dfe5f5")
cx = [X0 + i * (BW + GAP) + BW / 2 for i in range(len(P))]
busY = 165
S.add(f'<path d="M {rx + rw/2} 140 L {rx + rw/2} {busY} M {cx[0]} {busY} L {cx[-1]} {busY}" fill="none" stroke="{INK}" stroke-width="1.6"/>')
for i, (wid, name, own, acts) in enumerate(P):
    x = X0 + i * (BW + GAP)
    fill, edge = FILLS[own]
    S.add(f'<line x1="{cx[i]}" y1="{busY}" x2="{cx[i]}" y2="185" stroke="{INK}" stroke-width="1.6"/>')
    S.rect(x, 185, BW, 64, fill=fill, stroke=edge, sw=2, rx=6)
    S.text(cx[i], 210, f"{wid} {name}", 14.5, "bold")
    S.text(cx[i], 233, f"Owner: {own}", 12.5, fill=edge)
    S.add(f'<line x1="{cx[i]}" y1="249" x2="{cx[i]}" y2="275" stroke="{INK}" stroke-width="1.4"/>')
    bh = len(acts) * LH + 18
    S.rect(x, 275, BW, bh, fill="#ffffff", stroke="#b9c0cf", sw=1.2, rx=5)
    for j, a in enumerate(acts):
        S.text(x + 12, 298 + j * LH, a, 13, anchor="start")
LY = H - 30
for k, (own, lab) in enumerate([("Aditya", "Aditya: hardware, Pi software, integration, model"), ("Karan", "Karan: dataset and validation"), ("Both", "Both: management, thesis, presentations")]):
    fill, edge = FILLS[own]
    x = X0 + k * 520
    S.rect(x, LY - 14, 26, 18, fill=fill, stroke=edge, sw=2, rx=3)
    S.text(x + 36, LY, lab, 13.5, anchor="start")
open(os.path.join(OUT, "wbs.svg"), "w").write(S.svg())
print(W, H)
