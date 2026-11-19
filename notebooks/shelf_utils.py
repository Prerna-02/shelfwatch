"""Helpers for shelfwatch.ipynb: SKU-110K loading, dataset statistics, detection scoring, drawing,
and synthetic empty shelves (Part 4).
Boxes are x1, y1, x2, y2 in pixels unless stated. AP follows the same method as the course's det_utils.py.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))  # local repo layout; on Colab both files share a folder
from pipeline import shelf_rows  # noqa: E402


# ----------------------------------------------------------------- data
def split_images(root, split):
    """SKU-110K keeps all photos in one images/ folder; train.txt / val.txt / test.txt say which split each is in."""
    root = Path(root)
    return [str(root / line.strip()) for line in open(root / f"{split}.txt") if line.strip()]


def read_boxes(img_path, W, H):
    """YOLO label file (class cx cy w h, as fractions) -> (N, 4) pixel boxes."""
    p = Path(img_path).parent.parent / "labels" / (Path(img_path).stem + ".txt")
    if not p.exists():
        return np.zeros((0, 4))
    a = np.loadtxt(p, ndmin=2)[:, 1:5]
    cx, cy, w, h = a[:, 0] * W, a[:, 1] * H, a[:, 2] * W, a[:, 3] * H
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def dataset_stats(root, splits=("train", "val", "test"), grid=64):
    """One row per photo: split, products, width, height, shelf coverage (share of the photo covered by products).
    Plus every box's size relative to its photo and its centre. Reads image headers only, so it is fast."""
    rows, sizes, centres = [], [], []
    for split in splits:
        for p in split_images(root, split):
            W, H = Image.open(p).size
            b = read_boxes(p, W, H)
            g = np.zeros((grid, grid), bool)  # coverage on a coarse grid; overlapping boxes count once
            for x1, y1, x2, y2 in (b / [W, H, W, H] * grid).round().astype(int).clip(0, grid):
                g[y1:y2, x1:x2] = True
            rows.append((split, len(b), W, H, g.mean()))
            if len(b):
                bw, bh = (b[:, 2] - b[:, 0]) / W, (b[:, 3] - b[:, 1]) / H
                sizes.append(np.sqrt(bw * bh))
                centres.append(np.stack([(b[:, 0] + b[:, 2]) / 2 / W, (b[:, 1] + b[:, 3]) / 2 / H], 1))
    return rows, np.concatenate(sizes), np.concatenate(centres)


# ----------------------------------------------------------------- scoring
def iou_matrix(A, B):
    """IoU between every box in A (n, 4) and every box in B (m, 4) -> (n, m)."""
    ix1 = np.maximum(A[:, None, 0], B[None, :, 0]); iy1 = np.maximum(A[:, None, 1], B[None, :, 1])
    ix2 = np.minimum(A[:, None, 2], B[None, :, 2]); iy2 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda X: (X[:, 2] - X[:, 0]) * (X[:, 3] - X[:, 1])
    return inter / (area(A)[:, None] + area(B)[None, :] - inter + 1e-9)


def match(boxes, scores, gts, iou_thr=0.5):
    """Greedy matching, best score first, each real product used once. -> true-positive flag per prediction
    (in score order), and which real products were found."""
    order = np.argsort(-scores)
    tp, found = np.zeros(len(order), bool), np.zeros(len(gts), bool)
    if len(gts) and len(order):
        ious = iou_matrix(boxes[order], gts)
        for i in range(len(order)):
            row = np.where(found, -1, ious[i])
            j = row.argmax()
            if row[j] >= iou_thr:
                tp[i], found[j] = True, True
    return scores[order], tp, found


def average_precision(scores, tp, n_gt):
    """Area under the precision-recall curve (precision made monotonic, all points), as in det_utils.py."""
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    o = np.argsort(-scores)
    ctp = np.cumsum(tp[o]); rec, prec = ctp / n_gt, ctp / np.arange(1, len(o) + 1)
    mrec, mpre = np.r_[0, rec, rec[-1]], np.r_[1, prec, 0]
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def evaluate(preds, gts, conf=0.25):
    """preds: {photo: (boxes, scores)} made at a LOW threshold; gts: {photo: boxes}.
    -> dict with mAP@0.5, mAP@0.5:0.95, and precision, recall, count error at the app's threshold `conf`."""
    names = list(gts)
    n_gt = sum(len(gts[n]) for n in names)
    aps = []
    for t in np.arange(0.5, 0.96, 0.05):
        s, tp = zip(*[match(*preds[n], gts[n], t)[:2] for n in names])
        aps.append(average_precision(np.concatenate(s), np.concatenate(tp), n_gt))
    p, r = precision_recall(preds, gts, conf)
    count_err = np.array([abs((preds[n][1] >= conf).sum() - len(gts[n])) / max(len(gts[n]), 1) for n in names])
    return {"mAP@0.5": aps[0], "mAP@0.5:0.95": float(np.mean(aps)), "precision": p, "recall": r,
            "count error %": 100 * count_err.mean()}


def threshold_sweep(preds, gts, thresholds):
    """Precision, recall and F1 at each confidence threshold, to choose the app's threshold."""
    out = []
    for c in thresholds:
        p, r = precision_recall(preds, gts, c)
        out.append((c, p, r, 2 * p * r / max(p + r, 1e-9)))
    return np.array(out)


def precision_recall(preds, gts, conf):
    """Share of boxes that are real products, and share of real products found, at threshold `conf`."""
    tps = n_pred = n_gt = 0
    for n in gts:
        keep = preds[n][1] >= conf
        tps += match(preds[n][0][keep], preds[n][1][keep], gts[n])[1].sum()
        n_pred += keep.sum(); n_gt += len(gts[n])
    return tps / max(n_pred, 1), tps / max(n_gt, 1)


# ----------------------------------------------------------------- drawing
def draw(bgr, boxes, color=(0, 200, 0), thickness=3):
    """RGB copy of the photo with boxes drawn."""
    im = bgr.copy()
    for x1, y1, x2, y2 in np.asarray(boxes, int):
        cv2.rectangle(im, (x1, y1), (x2, y2), color, thickness)
    return im[:, :, ::-1]


def draw_errors(bgr, boxes, scores, gts, conf=0.25):
    """Green = product found, blue = false box, red = product missed. Returns RGB image and the three counts."""
    keep = scores >= conf
    b, s = boxes[keep], scores[keep]
    s_sorted, tp, found = match(b, s, gts)
    b = b[np.argsort(-s)]
    im = bgr.copy()
    for box, ok in zip(np.asarray(b, int), tp):
        cv2.rectangle(im, tuple(box[:2]), tuple(box[2:]), (0, 200, 0) if ok else (255, 120, 0), 3)
    for box in np.asarray(gts[~found], int):
        cv2.rectangle(im, tuple(box[:2]), tuple(box[2:]), (0, 0, 255), 4)
    return im[:, :, ::-1], int(tp.sum()), int((~tp).sum()), int((~found).sum())


# ----------------------------------------------------------------- Part 4: synthetic empty shelves
def plan_gap(gts, rng, n_remove=(1, 3)):
    """Choose 1-3 neighbouring products inside one shelf row to erase (never the first or last, so the gap stays
    inside the row). -> the true gap box (the area to erase), and indices of the products left untouched.
    Returns None when no row has enough products, or when overlapping neighbours leave too little room to erase."""
    rows = [r for r in shelf_rows(gts) if len(r) >= n_remove[1] + 2]
    if not rows:
        return None
    idx = rows[rng.integers(len(rows))]
    idx = idx[np.argsort(gts[idx, 0])]
    k = int(rng.integers(n_remove[0], n_remove[1] + 1))
    s = int(rng.integers(1, len(idx) - k))
    b = gts[idx[s:s + k]]
    pad = 0.05 * np.median(b[:, 3] - b[:, 1])  # a little margin so no edge of the product is left behind
    x1 = max(b[:, 0].min() - pad, gts[idx[s - 1], 2])  # ... but never into the neighbours
    x2 = min(b[:, 2].max() + pad, gts[idx[s + k], 0])
    if x2 - x1 < 0.8 * (b[:, 2].max() - b[:, 0].min()):  # neighbours overlap the products: they could not be fully erased
        return None
    gap = np.array([x1, b[:, 1].min() - pad, x2, b[:, 3].max() + pad])
    # "untouched" excludes anything the hole reaches into, so partly erased products do not count as missed later
    return gap, [i for i in range(len(gts)) if covered(gts[i], gap) < 0.02]


def covered(box, region):
    """Share of `box` that lies inside `region`."""
    w = max(0.0, min(box[2], region[2]) - max(box[0], region[0]))
    h = max(0.0, min(box[3], region[3]) - max(box[1], region[1]))
    return w * h / max((box[2] - box[0]) * (box[3] - box[1]), 1e-9)


def gap_mask(shape, gap):
    m = np.zeros(shape[:2], np.uint8)
    x1, y1, x2, y2 = np.round(gap).astype(int)
    m[max(y1, 0):y2, max(x1, 0):x2] = 255
    return m


def inpaint_classical(bgr, mask):
    """Baseline: OpenCV's Telea inpainting smears the surrounding pixels into the hole. No learning involved."""
    return cv2.inpaint(bgr, mask, 7, cv2.INPAINT_TELEA)


class LamaInpainter:
    """Generative baseline-beater: LaMa (Suvorov et al., WACV 2022), ONNX export by Carve (Apache-2.0).
    The model takes a fixed 512 x 512 input, so we crop a square around the hole with plenty of context,
    inpaint it, and paste back only the hole."""
    URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"

    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def __call__(self, bgr, mask):
        H, W = mask.shape
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return bgr.copy()
        side = int(min(H, W, max(np.ptp(ys), np.ptp(xs), 1) * 3))  # square: hole plus about one hole-width of context
        cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
        y0 = int(np.clip(cy - side // 2, 0, max(H - side, 0))); x0 = int(np.clip(cx - side // 2, 0, max(W - side, 0)))
        crop, m = bgr[y0:y0 + side, x0:x0 + side], mask[y0:y0 + side, x0:x0 + side]
        img = cv2.cvtColor(cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        m512 = (cv2.resize(m, (512, 512), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
        out = self.sess.run(None, {"image": img.transpose(2, 0, 1)[None].astype(np.float32) / 255,
                                   "mask": m512[None, None]})[0][0]
        if out.max() <= 1.0:  # this export returns 0-255; scale up if a different export returns 0-1
            out = out * 255
        out = cv2.cvtColor(out.transpose(1, 2, 0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        out = cv2.resize(out, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC)
        result = bgr.copy()
        result[y0:y0 + side, x0:x0 + side][m > 0] = out[m > 0]
        return result


def match_gaps(found, true_gap, thr=0.5):
    """A found gap matches the true one if it covers the same stretch of shelf: horizontal overlap of at least
    `thr` of the wider of the two, and some vertical overlap. -> (true gap found?, number of other gaps flagged)."""
    hits = 0
    for x1, y1, x2, y2, *_ in found:
        ox = min(x2, true_gap[2]) - max(x1, true_gap[0])
        oy = min(y2, true_gap[3]) - max(y1, true_gap[1])
        hits += ox >= thr * max(x2 - x1, true_gap[2] - true_gap[0]) and oy > 0
    return hits > 0, len(found) - min(hits, 1)
