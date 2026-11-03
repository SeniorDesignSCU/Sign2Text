import datetime as dt, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

OUT = os.path.dirname(os.path.abspath(__file__))
D = lambda m, d: dt.date(2026 if m >= 9 else 2027, m, d)
INK, RED, BLUE, GREEN, GREY, SHADE = "#1f2a44", "#c62828", "#2f5bd3", "#1b8a5a", "#7a8396", "#eceef3"
OWN = {"Aditya": BLUE, "Karan": GREEN, "Both": GREY}

# (id, label, start, end, owner, critical, kind)  kind: g=group, t=task, m=milestone
R = [
 ("1.0","Project management",D(11,2),D(5,28),"Both",0,"g"),
 ("1.1","Weekly reports (WARs)",D(11,2),D(5,28),"Both",0,"t"),
 ("1.3","Re-baseline schedule",D(11,2),D(11,8),"Aditya",0,"t"),
 ("1.5","Hazard + budget forms",D(11,2),D(11,8),"Both",0,"t"),
 ("1.6","Instructor meeting",D(11,2),D(11,15),"Both",0,"t"),
 ("2.0","Requirements & design",D(11,2),D(11,22),"Both",0,"g"),
 ("2.5","Final vocabulary",D(11,2),D(11,15),"Karan",0,"t"),
 ("2.6","ASL-fluent review",D(11,9),D(11,22),"Karan",0,"t"),
 ("3.0","Hardware",D(11,9),D(1,31),"Aditya",0,"g"),
 ("3.4","Camera mount",D(11,9),D(11,15),"Aditya",0,"t"),
 ("3.5","Speaker + amplifier",D(1,11),D(1,24),"Aditya",0,"t"),
 ("3.6","Portable build",D(1,25),D(1,31),"Aditya",0,"t"),
 ("3.7","Battery runtime test",D(1,25),D(1,31),"Karan",0,"t"),
 ("4.0","Device software",D(11,2),D(1,31),"Aditya",0,"g"),
 ("4.3","Browser recording mode",D(11,2),D(11,6),"Aditya",1,"t"),
 ("4.4","Recognizer integration",D(11,16),D(11,20),"Aditya",1,"t"),
 ("4.5","Sentence mode",D(1,18),D(1,24),"Aditya",0,"t"),
 ("4.6","Latency + idle tuning",D(1,18),D(1,31),"Aditya",1,"t"),
 ("4.7","Speech output",D(1,18),D(1,31),"Aditya",0,"t"),
 ("4.8","Boot-to-app",D(1,25),D(1,31),"Aditya",0,"t"),
 ("5.0","Dataset",D(11,2),D(2,11),"Karan",0,"g"),
 ("5.1","Recording protocol + storage",D(11,2),D(11,8),"Karan",0,"t"),
 ("5.3","Round 1 recording",D(11,7),D(11,12),"Karan",1,"t"),
 ("5.4","Recruit 3+ signers",D(11,9),D(12,4),"Karan",0,"t"),
 ("5.5","Round 2 recording",D(11,30),D(1,10),"Karan",1,"t"),
 ("5.7","Round 3 recording",D(2,8),D(2,11),"Karan",1,"t"),
 ("6.0","Model",D(11,2),D(2,14),"Aditya",0,"g"),
 ("6.1","Verify training pipeline",D(11,2),D(11,8),"Aditya",0,"t"),
 ("6.2","Model v1 + deploy",D(11,13),D(11,17),"Aditya",1,"t"),
 ("6.3","Retrain on round 2",D(1,11),D(1,17),"Aditya",1,"t"),
 ("6.4","Architecture freeze",D(1,17),D(1,17),"Aditya",1,"m"),
 ("6.5","Optimize for Pi",D(1,18),D(1,31),"Aditya",0,"t"),
 ("6.6","Final retrain",D(2,12),D(2,14),"Aditya",1,"t"),
 ("7.0","Testing & validation",D(11,9),D(2,28),"Karan",0,"g"),
 ("7.1","Test plan",D(11,9),D(11,15),"Karan",0,"t"),
 ("7.2","Robustness + thermal tests",D(11,16),D(11,22),"Karan",0,"t"),
 ("7.4","Accuracy test (fall)",D(11,30),D(12,4),"Karan",0,"t"),
 ("7.6","User testing round 1",D(2,1),D(2,7),"Karan",1,"t"),
 ("7.7","User testing round 2",D(2,15),D(2,21),"Karan",1,"t"),
 ("7.8","Results analysis",D(2,22),D(2,28),"Karan",1,"t"),
 ("8.0","Thesis",D(11,2),D(5,28),"Both",0,"g"),
 ("8.3","WBS + schedule",D(11,2),D(11,8),"Both",0,"t"),
 ("8.4","Interim report",D(11,16),D(12,4),"Both",0,"t"),
 ("8.5","Design + dataset chapters",D(1,4),D(2,28),"Both",0,"t"),
 ("8.8","Testing + results chapter",D(3,1),D(3,12),"Karan",1,"t"),
 ("8.9","Ethics + conclusions",D(3,29),D(4,18),"Both",0,"t"),
 ("8.10","Full draft + revisions",D(3,29),D(5,7),"Both",1,"t"),
 ("8.11","Final thesis (date TBD)",D(5,28),D(5,28),"Both",1,"m"),
 ("9.0","Presentations & demos",D(11,30),D(5,13),"Both",0,"g"),
 ("9.2","Fall demo",D(11,30),D(12,4),"Both",0,"t"),
 ("9.3","Demo video",D(2,22),D(2,28),"Both",0,"t"),
 ("9.4","Winter final demo",D(3,8),D(3,12),"Both",0,"t"),
 ("9.5","Poster + conference slides",D(4,19),D(5,9),"Both",0,"t"),
 ("9.7","Senior Design Conference (TBD)",D(5,13),D(5,13),"Both",0,"m"),
]
CHAIN = ["4.3","5.3","6.2","4.4","5.5","6.3","6.4","4.6","7.6","5.7","6.6","7.7","7.8","8.8","8.10","8.11"]
OFF = [(D(11,23),D(11,29),"Thanksgiving"), (D(12,7),D(1,3),"Finals + winter break"),
       (D(3,15),D(3,28),"Finals + spring break*")]

n = len(R)
fig, ax = plt.subplots(figsize=(17, 15.5), dpi=120)
x0, x1 = D(11,1), D(6,3)
num = mdates.date2num
for a, b, lab in OFF:
    ax.axvspan(num(a), num(b) + 1, color=SHADE, zorder=0)
    ax.text(num(a) + (num(b) - num(a) + 1) / 2, -0.85, lab, ha="center", va="center", fontsize=9.5, color=GREY, style="italic")
ypos = {}
for i, (wid, lab, s, e, own, crit, kind) in enumerate(R):
    y = i
    ypos[wid] = y
    if kind == "g":
        ax.add_patch(plt.Rectangle((num(s), y - 0.13), num(e) - num(s) + 1, 0.26, color=INK, zorder=3))
        for xx in (num(s), num(e) + 1):
            ax.plot([xx, xx], [y - 0.13, y + 0.33], color=INK, lw=2, zorder=3)
    elif kind == "m":
        ax.scatter([num(s) + 0.5], [y], marker="D", s=110, color=RED if crit else OWN[own], edgecolor=INK, zorder=5)
        ax.text(num(s) + 3.5, y, s.strftime("%b %-d"), va="center", fontsize=9, color=INK)
    else:
        col = OWN[own]
        ax.add_patch(plt.Rectangle((num(s), y - 0.32), num(e) - num(s) + 1, 0.64, facecolor=col,
                                   edgecolor=RED if crit else col, lw=2.6 if crit else 0.5, zorder=4))
ax.set_ylim(n - 0.4, -1.6)
ax.set_yticks(range(n))
ax.set_yticklabels([f"{w}  {l}" for (w, l, *_ ) in R], fontsize=10.5)
for tick, r in zip(ax.get_yticklabels(), R):
    if r[6] == "g":
        tick.set_fontweight("bold")
    if r[5]:
        tick.set_color(RED)

# dependency arrows along the critical path (finish -> start)
for a, b in zip(CHAIN, CHAIN[1:]):
    ra = next(r for r in R if r[0] == a); rb = next(r for r in R if r[0] == b)
    xa = num(ra[3]) + (0.5 if ra[6] == "m" else 1); ya = ypos[a]
    xb = num(rb[2]) + (0.5 if rb[6] == "m" else 0); yb = ypos[b]
    xm = max(xa + 0.8, min(xb - 0.6, xa + 2.5)) if xb > xa + 1.5 else xa + 0.8
    ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.3, shrinkA=0, shrinkB=2,
                                connectionstyle=f"angle,angleA=0,angleB=90,rad=0" if abs(yb - ya) > 0.1 else "arc3"),
                zorder=6)

ax.set_xlim(num(x0), num(x1))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
ax.tick_params(axis="x", which="major", labelsize=11, length=8)
ax.xaxis.tick_top()
ax.grid(axis="x", which="major", color="#c9ced9", lw=1)
ax.grid(axis="x", which="minor", color="#e6e9ef", lw=0.6)
for i, r in enumerate(R):
    if r[6] == "g" and i:
        ax.axhline(i - 0.5, color="#c9ced9", lw=0.8)
for s in ("right", "bottom", "left"):
    ax.spines[s].set_visible(False)
today = D(11,2)
ax.axvline(num(today), color=INK, lw=1.2, ls="--", zorder=2)
ax.text(num(today) + 0.6, n - 0.6, "today (Nov 2)", fontsize=9, color=INK)

ax.set_title("Signs to Speech — Gantt chart, Nov 2026 to May 2027", fontsize=17, fontweight="bold", pad=34, loc="left")
leg = [Patch(color=BLUE, label="Aditya"), Patch(color=GREEN, label="Karan"), Patch(color=GREY, label="Both"),
       Patch(facecolor="white", edgecolor=RED, lw=2.6, label="Critical path task"),
       Line2D([0], [0], color=RED, lw=1.3, marker=">", markersize=6, label="Dependency (finish → start)"),
       Line2D([0], [0], marker="D", color="w", markerfacecolor=RED, markeredgecolor=INK, markersize=9, label="Milestone"),
       Patch(color=SHADE, label="Non-working weeks"),
       Patch(color=INK, label="Work package (summary)")]
ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.42, -0.012), ncol=4, fontsize=10.5, frameon=False)
fig.text(0.01, -0.035, "*Spring break and the May dates are approximate; confirm against the 2026–27 SCU calendar and course schedule.",
         fontsize=9, color=GREY)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "gantt.png"), dpi=120, bbox_inches="tight", pad_inches=0.25)
print("ok", n, "rows")
