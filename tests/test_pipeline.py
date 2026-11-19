"""Tests for app/pipeline.py. Run from the shelfwatch folder: python -m pytest tests"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from pipeline import ProductDetector, analyse, blur_people, busyness, check_quality, find_gaps, gap_table, shelf_rows  # noqa: E402


class FakeSession:
    """Stands in for onnxruntime: returns a fixed (1, 5, N) YOLO output."""
    def __init__(self, cols):
        self.out = np.array(cols, np.float32).T[None]

    def run(self, _, feed):
        return [self.out]


def detector(cols):
    d = object.__new__(ProductDetector)  # skip loading a real .onnx file
    d.sess, d.size = FakeSession(cols), 960
    d.input = type("Input", (), {"name": "images"})
    return d


def test_box_maps_back_to_original_pixels():
    # 1920x1080 photo -> scale 0.5, grey padding of 210 px top and bottom.
    # Original box (100,200)-(300,600) -> letterboxed centre (100,410), size 100x200.
    d = detector([[100, 410, 100, 200, 0.9], [500, 500, 50, 50, 0.1]])
    boxes, scores = d.detect(np.zeros((1080, 1920, 3), np.uint8))
    assert np.allclose(boxes, [[100, 200, 300, 600]], atol=1) and np.allclose(scores, [0.9])


def test_duplicates_are_merged():
    d = detector([[100, 410, 100, 200, 0.9], [102, 411, 100, 200, 0.8]])
    boxes, _ = d.detect(np.zeros((1080, 1920, 3), np.uint8))
    assert len(boxes) == 1


def test_dense_shelf_keeps_every_product():
    # Regression: OpenCV's top_k once kept only 56 of 200 products. 200 products x 20 near-duplicates each.
    rng = np.random.default_rng(0)
    cols = []
    for k in range(200):
        x, y, s = 30 + (k % 20) * 45, 30 + (k // 20) * 90, rng.uniform(0.4, 0.95)  # all above the default threshold
        cols += [[x + rng.uniform(-1, 1), y + rng.uniform(-1, 1), 40, 80, s - rng.uniform(0, 0.02)] for _ in range(20)]
    boxes, _ = detector(cols).detect(np.zeros((960, 960, 3), np.uint8))
    assert len(boxes) == 200


def test_no_products_returns_empty():
    boxes, scores = detector([[100, 100, 50, 50, 0.1]]).detect(np.zeros((960, 960, 3), np.uint8))
    assert boxes.shape == (0, 4) and scores.shape == (0,)


def shelf(row_y, xs, w=40, h=80):
    """Boxes of width w standing on a shelf whose surface is at row_y, one per left edge in xs."""
    return [[x, row_y - h, x + w, row_y] for x in xs]


def test_rows_split_by_shelf_and_ignore_product_height():
    boxes = np.array(shelf(100, [0, 50]) + [[100, 40, 140, 102]] + shelf(300, [0, 50]), float)  # a taller product on row 1
    rows = shelf_rows(boxes)
    assert [sorted(r.tolist()) for r in rows] == [[0, 1, 2], [3, 4]]


def test_gap_between_products_and_at_shelf_end():
    # Row 1: products at 0..40, 50..90, then nothing until 250..290 -> one gap 90..250 (4 product widths).
    # Row 2: products 0..40 ... 200..240 (10 px spaces), then empty to the shelf end at 290 -> 50 px gap.
    boxes = np.array(shelf(100, [0, 50, 250]) + shelf(300, [0, 50, 100, 150, 200]), float)
    gaps, fullness = find_gaps(boxes)
    assert np.allclose(gaps[:, [0, 2, 4]], [[90, 250, 0], [240, 290, 1]])
    assert abs(fullness[0] - 120 / 290) < 1e-9 and abs(fullness[1] - 200 / 290) < 1e-9


def test_small_spaces_and_overlaps_are_not_gaps():
    # Two 10 px spaces (a quarter product each) and two overlapping boxes: no gap; empty space 20 of 175 px.
    boxes = np.array(shelf(100, [0, 50, 85, 135]), float)
    gaps, fullness = find_gaps(boxes)
    assert len(gaps) == 0 and abs(fullness[0] - 155 / 175) < 1e-9


def test_half_a_product_counts_as_a_gap():
    # The labelling rule: empty space of at least half a product width (20 px here) is a gap.
    gaps, _ = find_gaps(np.array(shelf(100, [0, 60, 100]), float))
    assert np.allclose(gaps[:, [0, 2]], [[40, 60]])


def test_no_products_means_no_gaps():
    gaps, fullness = find_gaps(np.zeros((0, 4)))
    assert gaps.shape == (0, 6) and fullness == []


def test_gap_width_is_reported_in_products():
    gaps, _ = find_gaps(np.array(shelf(100, [0, 50, 250]), float))  # 160 px gap, products 40 px wide
    assert gaps.shape == (1, 6) and abs(gaps[0, 5] - 4.0) < 1e-9


def test_detector_picks_the_requested_class():
    # A 2-class model: candidate A scores high only for class 1 (like "person" in COCO), B only for class 0.
    d = detector([[100, 100, 50, 50, 0.1, 0.9], [300, 300, 50, 50, 0.9, 0.1]])
    boxes, _ = d.detect(np.zeros((960, 960, 3), np.uint8), cls=1)
    assert len(boxes) == 1 and np.allclose(boxes[0], [75, 75, 125, 125])


def test_quality_check_flags_blurry_dark_and_small_photos():
    rng = np.random.default_rng(0)
    sharp = rng.integers(60, 200, (1000, 1000, 3), dtype=np.uint8)  # noisy texture = lots of detail
    assert check_quality(sharp) == []
    assert any("blurry" in p for p in check_quality(np.full((1000, 1000, 3), 120, np.uint8)))
    assert any("dark" in p for p in check_quality((sharp // 8).astype(np.uint8)))
    assert any("small" in p for p in check_quality(sharp[:300, :300]))


def test_people_are_pixelated_and_nothing_else_changes():
    bgr = np.random.default_rng(0).integers(0, 255, (200, 300, 3), dtype=np.uint8)
    out = blur_people(bgr, np.array([[50, 40, 150, 160]]))
    assert (out[:40] == bgr[:40]).all() and (out[:, 150:] == bgr[:, 150:]).all()
    assert not (out[40:160, 50:150] == bgr[40:160, 50:150]).all()
    assert len(np.unique(out[40:64, 50:74].reshape(-1, 3), axis=0)) == 1  # one 24 px block = one colour


class FakeDetector:
    def __init__(self, boxes, scores):
        self.out = (np.array(boxes, float).reshape(-1, 4), np.array(scores, float))

    def detect(self, bgr, conf=0.35, cls=0):
        return self.out


def test_analyse_ranks_gaps_and_flags_unsure_results():
    noisy = np.random.default_rng(0).integers(60, 200, (600, 600, 3), dtype=np.uint8)
    # Row 1 gap 90..250 = 4 products; row 2 ends at 240 while the shelf runs to 390 = 150 px = 3.75 products
    boxes = shelf(100, [0, 50, 250, 300, 350]) + shelf(300, [0, 50, 100, 150, 200])
    r = analyse(noisy, FakeDetector(boxes, [0.9] * 10), FakeDetector([[10, 10, 60, 90]], [0.8]))
    assert r["problems"] == [] and r["people"] == 1 and len(r["fullness"]) == 2
    table = gap_table(r["gaps"], 600)
    assert [g["width (products)"] for g in table] == [4.0, 3.8] and [g["shelf row (1 = top)"] for g in table] == [1, 2]
    assert not (r["image"][10:90, 10:60] == noisy[10:90, 10:60]).all()               # the person is blurred
    assert any("Fewer than 5" in p for p in analyse(noisy, FakeDetector(boxes[:3], [0.9] * 3))["problems"])
    assert any("unsure" in p for p in analyse(noisy, FakeDetector(boxes, [0.3] * 10))["problems"])


def test_gaps_behind_a_person_are_not_listed():
    noisy = np.random.default_rng(0).integers(60, 200, (600, 600, 3), dtype=np.uint8)
    boxes = shelf(100, [0, 50, 250, 300, 350]) + shelf(300, [0, 50, 100, 150, 200])  # gaps: row 1 at 90..250, row 2 at 240..390
    r = analyse(noisy, FakeDetector(boxes, [0.9] * 10), FakeDetector([[230, 200, 400, 320]], [0.8]))  # person over row 2's gap
    assert len(r["gaps"]) == 1 and r["gaps"][0][4] == 0
    assert any("behind people" in p for p in r["problems"])


def textured_shelf(empty_at):
    """A grey shelf photo: busy random texture inside every product box, plain grey where `empty_at` is."""
    im = np.full((400, 600, 3), 110, np.uint8)
    rng = np.random.default_rng(1)
    boxes = shelf(200, [0, 50, 100, 150, 200, 250, 450, 500, 550]) + shelf(380, [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550])
    for x1, y1, x2, y2 in boxes:
        im[y1:y2, x1:x2] = rng.integers(0, 255, (y2 - y1, x2 - x1, 3), dtype=np.uint8)
    return im, np.array(boxes, float)


def test_busyness_tells_empty_shelf_from_a_missed_product():
    im, boxes = textured_shelf(None)
    gaps = np.array([[290, 120, 450, 200, 0, 4.0]])                                # plain grey: a real empty shelf
    assert busyness(im, gaps, boxes)[0] < 0.3
    missed = boxes[[2]]                                                             # a product's own texture
    assert busyness(im, np.c_[missed, [[1, 1.0]]], boxes)[0] > 0.8
    assert busyness(im, np.zeros((0, 6)), boxes).shape == (0,)


def test_analyse_lists_empty_gaps_before_busy_ones():
    im, boxes = textured_shelf(None)
    hidden_product = boxes[7]                       # pretend the detector missed this product: a 'gap' with texture
    kept = np.delete(boxes, 7, axis=0)
    r = analyse(im, FakeDetector(kept, [0.9] * len(kept)))
    assert list(r["empty"]) == sorted(r["empty"], reverse=True)                     # empty ones first
    table = gap_table(r["gaps"], 600, r["empty"])
    assert table[0]["status"] == "empty shelf: restock" and table[0]["width (products)"] == 4.0
    busy = [g for g, e in zip(r["gaps"], r["empty"]) if not e]
    assert any(g[0] <= hidden_product[0] + 5 and g[2] >= hidden_product[2] - 5 for g in busy)
