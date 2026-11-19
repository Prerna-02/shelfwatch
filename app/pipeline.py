"""ShelfWatch image pipeline: quality check -> blur people -> detect products -> rows -> gaps.

No Streamlit code here, so the notebook and the tests use exactly the same functions as the app.
Everything the app needs: quality check, product detection, people blurring, shelf rows, gaps and the restock list.
"""
import cv2
import numpy as np
import onnxruntime as ort

CONF = 0.35     # box confidence threshold, chosen on validation photos in shelfwatch.ipynb Step 3.1
MIN_GAP = 0.5   # a gap is empty space at least half a product wide: the business rule in docs/labelling_guide.md
EMPTY_CUT = 0.3  # a gap with under 30% of a typical product's edges looks empty; tuned on the odd-numbered own photos,
                 # tested on the even ones (shelfwatch.ipynb Step 5.3)


def letterbox(bgr, size):
    """Resize keeping the aspect ratio, pad to size x size with grey (what YOLO was trained on)."""
    h, w = bgr.shape[:2]
    s = size / max(h, w)
    nw, nh = round(w * s), round(h * s)
    px, py = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[py:py + nh, px:px + nw] = cv2.resize(bgr, (nw, nh))
    return out, s, px, py


class ProductDetector:
    """A YOLO11 model exported to ONNX, run on CPU with onnxruntime. Used for the fine-tuned shelf model (one class:
    product) and for the COCO model the app uses to find people (class 0: person)."""

    def __init__(self, onnx_path, size=None):
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0]
        self.size = size or self.input.shape[2]  # fixed at export (960) unless the model accepts any size

    def detect(self, bgr, conf=CONF, iou=0.7, max_det=1000, cls=0):
        """Boxes of class `cls` -> (N, 4) as x1, y1, x2, y2 in original-photo pixels, and scores (N,), best first."""
        im, s, px, py = letterbox(bgr, self.size)
        x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        out = self.sess.run(None, {self.input.name: x})[0][0]  # (4 + classes, N): cx, cy, w, h, one score per class
        if out.shape[0] < 5 + cls:
            raise ValueError(f"model has {out.shape[0] - 4} classes, asked for class {cls}")
        keep = out[4 + cls] >= conf
        cx, cy, w, h = out[:4, keep]
        sc = out[4 + cls, keep]
        xywh = np.stack([(cx - w / 2 - px) / s, (cy - h / 2 - py) / s, w / s, h / s], 1)
        # top_k stays 0: OpenCV's top_k trims candidates BEFORE NMS, which silently dropped products on dense shelves
        idx = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), sc.tolist(), conf, iou), int).reshape(-1)[:max_det]
        boxes = xywh[idx].copy()
        boxes[:, 2:] += boxes[:, :2]
        H, W = bgr.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
        return boxes, sc[idx]


def shelf_rows(boxes, tol=0.5):
    """Group product boxes into shelf rows, top to bottom. Products on one shelf stand on the same surface, so a
    new row starts where bottom edges jump by more than `tol` x the median product height. -> list of index arrays."""
    if len(boxes) == 0:
        return []
    height = np.median(boxes[:, 3] - boxes[:, 1])
    order = np.argsort(boxes[:, 3])
    breaks = np.where(np.diff(boxes[order, 3]) > tol * height)[0] + 1
    return np.split(order, breaks)


def find_gaps(boxes, min_gap=MIN_GAP, tol=0.5):
    """Empty shelf space in each row: wider than `min_gap` x the row's median product width.
    A row's shelf is taken to span as far as the products on it and on the rows directly above and below, so empty
    shelf ends are found without a short row (end cap, far end of an angled photo) borrowing a distant row's width.
    -> gaps (N, 6) as x1, y1, x2, y2, row, width in products; fullness per row (share of the row covered by products)."""
    gaps, fullness = [], []
    rows = shelf_rows(boxes, tol)
    for r, idx in enumerate(rows):
        near = np.concatenate(rows[max(r - 1, 0):r + 2])
        left, right = boxes[near, 0].min(), boxes[near, 2].max()
        b = boxes[idx][np.argsort(boxes[idx, 0])]
        width = np.median(b[:, 2] - b[:, 0])
        top, bottom = np.median(b[:, 1]), np.median(b[:, 3])
        starts = np.r_[left, np.maximum.accumulate(b[:, 2])]  # furthest right edge so far (boxes can overlap)
        ends = np.r_[b[:, 0], right]
        empty = np.clip(ends - starts, 0, None)
        fullness.append(float(1 - empty.sum() / max(right - left, 1e-9)))
        gaps += [(s, top, e, bottom, r, w / width) for s, e, w in zip(starts, ends, empty) if w >= min_gap * width]
    return np.array(gaps, float).reshape(-1, 6), fullness


def check_quality(bgr):
    """Reasons the photo may give an unreliable result (empty list = fine). Thresholds are deliberately loose:
    the sharpest-looking of our own 40 shelf photos scored 840+ on sharpness, the darkest 83 on brightness."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    problems = []
    if min(h, w) < 480:
        problems.append("The photo is small (under 480 pixels). Take it closer, or upload the original file.")
    if cv2.Laplacian(cv2.resize(gray, (800, max(1, round(800 * h / w)))), cv2.CV_64F).var() < 150:
        problems.append("The photo looks blurry. Hold the phone still and retake it.")
    if gray.mean() < 40:
        problems.append("The photo is very dark. Turn on more light or retake it.")
    return problems


def blur_people(bgr, boxes):
    """Copy of the photo with every person box pixelated beyond recognition (faces included)."""
    out = bgr.copy()
    for x1, y1, x2, y2 in np.asarray(boxes, int):
        roi = out[y1:y2, x1:x2]
        if roi.size:
            small = cv2.resize(roi, (max(1, roi.shape[1] // 24), max(1, roi.shape[0] // 24)), interpolation=cv2.INTER_AREA)
            out[y1:y2, x1:x2] = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    return out


def _overlap(a, b):
    """Share of box a's area that box b covers."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h / max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)


def busyness(bgr, gaps, boxes):
    """How 'busy' each gap looks: its share of edge pixels divided by a typical product's (median over the photo).
    An empty shelf back is plain (low); a product the detector missed has print and outlines (high). Edges are found
    at a fixed 1000 px width so the measure does not depend on photo size; boxes are shrunk 15% to skip borders."""
    if not len(gaps):
        return np.zeros(0)
    H, W = bgr.shape[:2]
    s = 1000 / W
    gray = cv2.cvtColor(cv2.resize(bgr, (1000, max(1, round(H * s)))), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150) > 0

    def density(box):
        x1, y1, x2, y2 = np.asarray(box[:4], float) * s
        dx, dy = (x2 - x1) * 0.15, (y2 - y1) * 0.15
        r = edges[int(y1 + dy):int(y2 - dy), int(x1 + dx):int(x2 - dx)]
        return r.mean() if r.size else 1.0

    typical = np.median([density(b) for b in boxes]) if len(boxes) else 1.0
    return np.array([density(g) for g in gaps]) / max(typical, 1e-9)


def analyse(bgr, products, people=None, tol=0.5):
    """Everything the app shows for one shelf photo. `products` / `people` are ProductDetector instances.
    -> dict: image (people blurred), boxes, scores, gaps, empty (per gap: passes the emptiness check), fullness per row,
    people count, problems. Gaps that look empty come first, each group largest first."""
    problems = check_quality(bgr)
    boxes, scores = products.detect(bgr)
    gaps, fullness = find_gaps(boxes, tol=tol)
    persons = people.detect(bgr, conf=0.25, cls=0)[0] if people else np.zeros((0, 4))
    hidden = [any(_overlap(g, p) > 0.3 for p in persons) for g in gaps]  # a person in front hides the shelf, not empties it
    if any(hidden):
        gaps = gaps[~np.array(hidden)]
        n = sum(hidden)
        problems.append(f"{n} spot{'s' if n > 1 else ''} hidden behind people {'are' if n > 1 else 'is'} not listed. "
                        "Retake the photo without people in front of the shelf to check.")
    empty = busyness(bgr, gaps, boxes) < EMPTY_CUT
    order = np.lexsort((-gaps[:, 5], ~empty)) if len(gaps) else np.zeros(0, int)
    gaps, empty = gaps[order], empty[order]
    if len(boxes) < 5:
        problems.append("Fewer than 5 products found: this may not be a shelf, or the shelf is too far away.")
    elif scores.mean() < 0.5:
        problems.append(f"The model is unsure about many products (average confidence {scores.mean():.2f}). "
                        "Check the result by eye before acting on it.")
    return {"image": blur_people(bgr, persons), "boxes": boxes, "scores": scores, "gaps": gaps, "empty": empty,
            "fullness": fullness, "people": len(persons), "problems": problems}


def gap_table(gaps, width, empty=None):
    """Restock list, in the order given: one dict per gap. `width` is the photo width, for the position."""
    empty = np.ones(len(gaps), bool) if empty is None else empty
    return [{"priority": i + 1, "status": "empty shelf: restock" if e else "check: may be a missed product",
             "shelf row (1 = top)": int(g[4]) + 1, "position from left (%)": round(100 * (g[0] + g[2]) / 2 / width),
             "width (products)": round(float(g[5]), 1), "x1": int(g[0]), "y1": int(g[1]), "x2": int(g[2]), "y2": int(g[3])}
            for i, (g, e) in enumerate(zip(gaps, empty))]


def draw_result(bgr, boxes, gaps, empty=None):
    """Products in green; gaps numbered by priority, red if they look empty, orange if they need a check. BGR copy."""
    im = bgr.copy()
    t = max(2, im.shape[1] // 400)
    empty = np.ones(len(gaps), bool) if empty is None else empty
    for x1, y1, x2, y2 in np.asarray(boxes, int):
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 200, 0), t)
    for i, (g, e) in enumerate(zip(gaps, empty)):
        x1, y1, x2, y2 = g[:4].astype(int)
        color = (0, 0, 255) if e else (0, 165, 255)
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2 * t if e else t)
        cv2.putText(im, str(i + 1), (x1 + 4 * t, y1 + 16 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * t, color, 2 * t)
    return im
