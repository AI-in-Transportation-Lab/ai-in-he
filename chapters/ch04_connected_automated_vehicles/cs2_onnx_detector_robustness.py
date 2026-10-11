"""Chapter 4 - Case Study 2: Deploying an Object Detector at the Edge and Testing Its Robustness.

Book: Artificial Intelligence in Highway Engineering
Source: chapters/04a.tex (2 listing block(s), concatenated in book order).
Run from the repository root with `make ch04` or `python scripts/run_all.py --chapter 4`;
figures are written to outputs/ch04/.
"""
# Files used (download once):
#   curl -L -o yolo11n.onnx \
#     https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx
#   curl -L -o traffic_scene.jpg \
#     https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg
import time
import cv2
import numpy as np
import onnxruntime as ort

COCO = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
        "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
        "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
        "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
        "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
        "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
        "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
        "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
        "remote","keyboard","cell phone","microwave","oven","toaster","sink",
        "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
        "toothbrush"]
ROAD_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck",
                "traffic light", "stop sign"}

class Detector:
    """YOLO11n exported to ONNX, run with ONNX Runtime on CPU."""
    def __init__(self, path="yolo11n.onnx", size=640, conf=0.25, iou=0.45):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.size, self.conf, self.iou = size, conf, iou

    def preprocess(self, bgr):
        """Letterbox to size x size, BGR->RGB, HWC->NCHW, scale to [0, 1]."""
        h, w = bgr.shape[:2]
        r = self.size / max(h, w)
        nh, nw = round(h * r), round(w * r)
        pad_y, pad_x = (self.size - nh) // 2, (self.size - nw) // 2
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = cv2.resize(bgr, (nw, nh))
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(x), r, pad_x, pad_y

    def __call__(self, bgr):
        x, r, px, py = self.preprocess(bgr)
        out = self.sess.run(None, {self.inp: x})[0][0].T     # (anchors, 4 + 80)
        boxes, scores = out[:, :4], out[:, 4:]
        cls = scores.argmax(1); conf = scores.max(1)
        keep = conf >= self.conf
        boxes, cls, conf = boxes[keep], cls[keep], conf[keep]
        # (cx, cy, w, h) in letterboxed pixels -> (x1, y1, x2, y2) in original pixels
        xyxy = np.column_stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                                boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2])
        xyxy = (xyxy - [px, py, px, py]) / r
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, bgr.shape[1] - 1)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, bgr.shape[0] - 1)
        # class-aware NMS: offset boxes by class so classes never suppress each other
        off = cls[:, None] * 4096.0
        nms_boxes = [[float(a), float(b), float(c - a), float(d - b)]
                     for a, b, c, d in (xyxy + off)]
        idx = cv2.dnn.NMSBoxes(nms_boxes, conf.tolist(), self.conf, self.iou)
        idx = np.array(idx).flatten() if len(idx) else np.array([], int)
        return [(COCO[cls[i]], float(conf[i]), xyxy[i].round().astype(int).tolist())
                for i in idx if COCO[cls[i]] in ROAD_CLASSES]

det = Detector()
frame = cv2.imread("traffic_scene.jpg")
detections = sorted(det(frame), key=lambda d: -d[1])
print(f"Frame {frame.shape[1]}x{frame.shape[0]}; {len(detections)} road-relevant detections")
for name, c, (x1, y1, x2, y2) in detections:
    print(f"  {name:<8} conf {c:.2f}  box [{x1:4d},{y1:4d},{x2:4d},{y2:4d}]")

times = []
for _ in range(20):
    t0 = time.perf_counter(); det(frame); times.append(time.perf_counter() - t0)
print(f"CPU latency (median of 20 runs, incl. pre/post-processing): "
      f"{1000 * np.median(times):.0f} ms")


# ---------- Robustness to simulated adverse conditions ----------
def fog(img, a=0.55):   return cv2.addWeighted(img, 1 - a, np.full_like(img, 200), a, 0)
def low_light(img, g=2.5):
    return (255 * (img / 255.0) ** g).astype(np.uint8)
def noise(img, sd=25, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0, sd, img.shape), 0, 255).astype(np.uint8)
def motion_blur(img, k=21):
    kernel = np.zeros((k, k), np.float32); kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)

def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

def retained(ref, test, thr=0.5):
    """Fraction of reference detections matched (same class, IoU >= thr), greedy."""
    used, confs = set(), []
    for name, _, rb in ref:
        best, bj = 0.0, None
        for j, (n2, c2, tb) in enumerate(test):
            if j not in used and n2 == name and box_iou(rb, tb) > best:
                best, bj = box_iou(rb, tb), j
        if bj is not None and best >= thr:
            used.add(bj); confs.append(test[bj][1])
    return len(confs) / len(ref), (np.mean(confs) if confs else float("nan"))

ref = detections
sweeps = {"fog (blend weight)": (fog, [0.5, 0.7, 0.8, 0.9]),
          "low light (gamma)": (low_light, [2.0, 3.0, 4.0, 5.0]),
          "sensor noise (s.d.)": (noise, [20, 40, 60, 80]),
          "motion blur (kernel px)": (motion_blur, [11, 31, 51, 71])}
print(f"\n{'Degradation':<24}{'Level':>6}{'Dets':>6}{'Retained':>10}{'Mean conf':>11}")
results = {}
for label, (fn, levels) in sweeps.items():
    results[label] = []
    for lv in levels:
        d = det(fn(frame, lv))
        r, mc = retained(ref, d)
        results[label].append((lv, r, mc))
        print(f"{label:<24}{lv:>6}{len(d):>6}{r:>10.2f}{mc:>11.2f}")

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
vis = frame.copy()
for name, c, (x1, y1, x2, y2) in ref:
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 180, 218), 4)  # gold (BGR)
    txt = f"{name} {c:.2f}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    lx = min(x1, vis.shape[1] - tw - 10)        # keep label inside frame
    ly = max(y1, th + 14)                       # label sits above the box
    cv2.rectangle(vis, (lx, ly - th - 14), (lx + tw + 10, ly),
                  (69, 74, 63), -1)             # dark label background
    cv2.putText(vis, txt, (lx + 5, ly - 7), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2)        # white score text
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2),
                       gridspec_kw={"width_ratios": [1, 1.6]})
ax[0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); ax[0].axis("off")
ax[0].set_title("(a) Detections on the clean frame", fontsize=9)
for (label, rows), col in zip(results.items(), [GOLD, SAGE, SKY, RUST]):
    ax[1].plot(range(1, 5), [r for _, r, _ in rows], "o-", color=col,
               lw=2, ms=6, label=label)
ax[1].set(xticks=range(1, 5),
          xticklabels=["mild", "moderate", "strong", "severe"],
          ylim=(-0.05, 1.05), ylabel="Retained fraction of clean detections",
          title="(b) Robustness to simulated degradations")
ax[1].legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig("chap04_fig_detection_robustness.png", dpi=200)
