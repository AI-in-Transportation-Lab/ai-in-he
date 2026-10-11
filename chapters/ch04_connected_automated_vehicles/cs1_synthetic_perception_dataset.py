"""Chapter 4 - Case Study 1: A Synthetic, Automatically Labeled Perception Dataset.

Book: Artificial Intelligence in Highway Engineering
Source: chapters/04a.tex (2 listing block(s), concatenated in book order).
Run from the repository root with `make ch04` or `python scripts/run_all.py --chapter 4`;
figures are written to outputs/ch04/.
"""
import json
import cv2
import numpy as np

W, H = 800, 600
ROAD_L, ROAD_R, N_LANES = 200, 600, 3
LANE_W = (ROAD_R - ROAD_L) / N_LANES
CLASSES = {"car": 0, "truck": 1}
SIZE = {"car": ((40, 56), (80, 110)), "truck": ((60, 72), (140, 190))}
MIN_GAP = 20                     # minimum longitudinal gap between vehicles (px)

def draw_road(rng):
    """Road surface with randomized tone, solid edges, and dashed lane lines."""
    img = np.full((H, W, 3), (40, 90, 40), np.uint8)          # grass verge
    tone = int(rng.integers(60, 100))
    img[:, ROAD_L:ROAD_R] = (tone, tone, tone)
    for x in (ROAD_L, ROAD_R):
        cv2.line(img, (x, 0), (x, H), (235, 235, 235), 4)
    offset = int(rng.integers(0, 80))                          # dash phase
    for k in range(1, N_LANES):
        x = int(ROAD_L + k * LANE_W)
        for y in range(-offset, H, 80):
            cv2.line(img, (x, max(y, 0)), (x, min(y + 45, H)), (220, 220, 220), 3)
    return img

def place_vehicles(rng):
    """Sample non-overlapping vehicles lane by lane; allow truncation at edges."""
    objects = []
    for lane in range(N_LANES):
        cx = ROAD_L + (lane + 0.5) * LANE_W
        y = int(rng.integers(-40, 60))                         # may start above frame
        while True:
            cls = "truck" if rng.random() < 0.2 else "car"
            (wmin, wmax), (hmin, hmax) = SIZE[cls]
            w, h = int(rng.integers(wmin, wmax)), int(rng.integers(hmin, hmax))
            if y + 0.7 * h > H:                                # at most 30% cut off
                break
            if rng.random() < 0.55:                            # occupy this slot
                x1 = int(cx - w / 2 + rng.integers(-8, 9))
                objects.append((cls, x1, y, x1 + w, y + h))
            y += h + MIN_GAP + int(rng.integers(0, 120))
    return objects

def apply_condition(img, cond, rng):
    """Simple photometric domain randomization."""
    out = img.astype(np.float32)
    if cond == "night":
        out *= 0.35
    elif cond == "fog":
        out = 0.45 * out + 0.55 * 200
    out += rng.normal(0, 6 if cond == "clear" else 10, out.shape)   # sensor noise
    return np.clip(out, 0, 255).astype(np.uint8)

def render_scene(idx, rng):
    img = draw_road(rng)
    cond = rng.choice(["clear", "night", "fog"], p=[0.5, 0.25, 0.25])
    labels = []
    for cls, x1, y1, x2, y2 in place_vehicles(rng):
        color = tuple(int(c) for c in rng.integers(30, 256, 3))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(img, (x1 + 6, y1 + 12), (x2 - 6, y1 + 30), (30, 30, 30), -1)  # windscreen
        bx1, by1, bx2, by2 = max(x1, 0), max(y1, 0), min(x2, W - 1), min(y2, H - 1)
        truncated = (bx1, by1, bx2, by2) != (x1, y1, x2, y2)
        labels.append({"class": cls, "bbox": [bx1, by1, bx2, by2],
                       "truncated": bool(truncated)})
    img = apply_condition(img, cond, rng)
    return img, {"file": f"synthetic_{idx:03d}.png", "condition": str(cond),
                 "objects": labels}

def to_yolo(obj):
    x1, y1, x2, y2 = obj["bbox"]
    return (f"{CLASSES[obj['class']]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
            f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")

rng = np.random.default_rng(42)
dataset = []
for i in range(200):
    img, meta = render_scene(i, rng)
    cv2.imwrite(meta["file"], img)
    with open(meta["file"].replace(".png", ".txt"), "w") as f:
        f.write("\n".join(to_yolo(o) for o in meta["objects"]))
    dataset.append(meta)
with open("annotations.json", "w") as f:
    json.dump(dataset, f, indent=1)

# ---- Dataset audit ----
objs = [o for d in dataset for o in d["objects"]]
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
max_iou = max((iou(p["bbox"], q["bbox"]) for d in dataset
               for i, p in enumerate(d["objects"]) for q in d["objects"][i + 1:]),
              default=0.0)
inside = all(0 <= o["bbox"][0] < o["bbox"][2] < W and 0 <= o["bbox"][1] < o["bbox"][3] < H
             for o in objs)
cond = {c: sum(d["condition"] == c for d in dataset) for c in ("clear", "night", "fog")}
print(f"Images: {len(dataset)}; objects: {len(objs)} "
      f"(car {sum(o['class'] == 'car' for o in objs)}, "
      f"truck {sum(o['class'] == 'truck' for o in objs)}); "
      f"truncated: {sum(o['truncated'] for o in objs)}")
print(f"Objects per image: mean {len(objs) / len(dataset):.2f}, "
      f"min {min(len(d['objects']) for d in dataset)}, "
      f"max {max(len(d['objects']) for d in dataset)}")
print(f"Conditions: {cond}")
print(f"All boxes inside image: {inside}; maximum pairwise IoU: {max_iou:.3f}")
print(f"First image ({dataset[0]['condition']}):")
for o in dataset[0]["objects"]:
    print(f"  {o['class']:<6}{str(o['bbox']):<24}truncated={o['truncated']}")
print("YOLO label line:", to_yolo(dataset[0]["objects"][0]))


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GOLD, SAGE, SKY = "#DAB400", "#78978D", "#6ADBF1"  # Moonrise Kingdom palette
RUST, OLIVE = "#CE6B08", "#A59D30"
INK, PAPER, GRID = "#3F4A45", "#FBF8F4", "#D5D5D3" # text, background, grid
plt.rcParams.update({
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": SAGE,
    "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
picks = []
for c in ("clear", "night", "fog"):
    picks.append(next(d for d in dataset if d["condition"] == c))
picks.append(max(dataset, key=lambda d: len(d["objects"])))
fig, ax = plt.subplots(1, 4, figsize=(12, 2.9))
for a, d in zip(ax, picks):
    img = cv2.cvtColor(cv2.imread(d["file"]), cv2.COLOR_BGR2RGB)
    for o in d["objects"]:
        x1, y1, x2, y2 = o["bbox"]
        col = (206, 107, 8) if o["class"] == "truck" else (106, 219, 241)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
    a.imshow(img[:, 150:650]); a.axis("off")
    n_obj = len(d["objects"])
    a.set_title(f"{d['file']} ({d['condition']}, {n_obj} obj.)", fontsize=8)
fig.tight_layout()
fig.savefig("chap04_fig_synthetic_samples.png", dpi=200)
