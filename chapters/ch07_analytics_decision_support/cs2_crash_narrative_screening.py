"""Chapter 7 - Case Study 2: Evidence Screening in Crash Narratives.

Book: Artificial Intelligence in Highway Engineering
Source: chapters/07a.tex (3 listing block(s), concatenated in book order).
Run from the repository root with `make ch07` or `python scripts/run_all.py --chapter 7`;
figures are written to outputs/ch07/.
"""
narratives = [
    "Alcohol was a factor according to the investigating officer.",
    "There was no evidence of alcohol involvement.",
    "The alcohol test was pending at the time of the report.",
    "The vehicle skidded on wet pavement and struck a tree.",
    "Driver 2 admitted drinking alcohol before driving.",
    "Alcohol involvement was ruled out by the investigator.",
    "The driver showed slurred speech; the cause was undetermined.",
    "An alcohol container was found in the passenger compartment.",
    "The truck crossed the centerline while overtaking.",
    "Alcohol was a factor in a previous crash, not this event.",
]
reference = [
    "affirmed", "negated", "unresolved", "not mentioned",
    "affirmed", "negated", "not mentioned", "unresolved",
    "not mentioned", "negated",
]


import re

patterns = [
    ("negated", r"no evidence of alcohol involvement|"
                r"alcohol involvement was ruled out"),
    ("unresolved", r"alcohol test (?:was )?pending"),
    ("affirmed", r"alcohol was a factor|"
                 r"driver 2 admitted drinking alcohol"),
]

def screen(text):
    for label, pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            evidence = text[match.start():match.end()]
            return label, evidence
    match = re.search(r"\balcohol\b", text, re.I)
    if match:
        return "unresolved", text[match.start():match.end()]
    return "not mentioned", ""

predictions = [screen(text)[0] for text in narratives]
for i, (truth, pred) in enumerate(
        zip(reference, predictions), start=1):
    print(f"T{i:02d}: reference={truth}; predicted={pred}")

pos = "affirmed"
tp = sum(t == pos and p == pos for t, p in zip(reference, predictions))
fp = sum(t != pos and p == pos for t, p in zip(reference, predictions))
fn = sum(t == pos and p != pos for t, p in zip(reference, predictions))
tn = len(reference) - tp - fp - fn
exact = sum(t == p for t, p in zip(reference, predictions))
print(f"\nBinary ('affirmed'): TP={tp} FP={fp} FN={fn} TN={tn}")
print(f"Precision={tp/(tp+fp):.3f} Recall={tp/(tp+fn):.3f} "
      f"Binary accuracy={(tp+tn)/len(reference):.2f}")
print(f"Four-class agreement: {exact}/{len(reference)}")


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

GOLD, SAGE, SKY = "#DAB400", "#78978D", "#6ADBF1"  # Moonrise Kingdom palette
RUST, OLIVE = "#CE6B08", "#A59D30"
INK, PAPER, GRID = "#3F4A45", "#FBF8F4", "#D5D5D3" # text, background, grid
plt.rcParams.update({
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": SAGE,
    "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
labels = ["affirmed", "negated", "unresolved", "not mentioned"]
cm = [[sum(t == r and p == c for t, p in zip(reference, predictions))
       for c in labels] for r in labels]
fig, ax = plt.subplots(figsize=(5.6, 4.4))
ax.imshow(cm, vmin=0, vmax=3,
          cmap=LinearSegmentedColormap.from_list("gold", [PAPER, GOLD]))
for i in range(4):
    for j in range(4):
        ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=12)
        if i != j and cm[i][j] > 0:                  # outline the errors
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor=RUST, lw=2.5))
ax.set(xticks=range(4), xticklabels=labels, yticks=range(4),
       yticklabels=labels, xlabel="Screen output", ylabel="Reference label",
       title=f"Four-class agreement: {exact} of {len(reference)}")
ax.grid(False)
ax.spines[:].set_visible(False)                   # no frame on a matrix
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig("chap07_fig02_narrative_screen.png", dpi=200)
