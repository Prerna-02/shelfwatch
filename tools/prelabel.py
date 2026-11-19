"""Pre-label shelf photos with the trained model, so labelling means correcting boxes, not drawing ~100 per photo.

Usage (from the shelfwatch folder):  .venv/Scripts/python tools/prelabel.py data/own_shelves
Writes   <folder>/prelabels/<photo>.txt   YOLO format: class cx cy w h (fractions); class 0 = product
         <folder>/prelabels/labels.txt    class names for the labelling tool: product, gap
         <folder>/prelabel_preview/       each photo with its boxes drawn, to judge quality quickly
Hand-corrected labels belong in <folder>/labels/, which this script never writes to.
Gaps are never pre-labelled: you draw every gap box yourself, because that is what the app is tested on.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from pipeline import CONF, ProductDetector  # noqa: E402

MODEL = Path(__file__).resolve().parents[1] / "models" / "shelf_yolo11n_960.onnx"


def main(folder):
    folder = Path(folder)
    photos = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not photos:
        sys.exit(f"no photos found in {folder}")
    labels, preview = folder / "prelabels", folder / "prelabel_preview"
    labels.mkdir(exist_ok=True); preview.mkdir(exist_ok=True)
    (labels / "labels.txt").write_text("product\ngap\n")
    det = ProductDetector(str(MODEL))
    for p in photos:
        bgr = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)  # imdecode handles any path on Windows
        H, W = bgr.shape[:2]
        boxes, scores = det.detect(bgr, conf=CONF)
        with open(labels / f"{p.stem}.txt", "w") as f:
            for x1, y1, x2, y2 in boxes:
                f.write(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
        for x1, y1, x2, y2 in boxes.astype(int):
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 200, 0), max(2, W // 500))
        cv2.imwrite(str(preview / f"{p.stem}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"{p.name}: {len(boxes)} products, mean confidence {scores.mean() if len(scores) else 0:.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/own_shelves")
