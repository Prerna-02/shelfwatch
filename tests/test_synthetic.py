"""Tests for the Part 4 helpers in notebooks/shelf_utils.py (synthetic empty shelves). Run: python -m pytest tests"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
from shelf_utils import LamaInpainter, covered, gap_mask, match_gaps, plan_gap  # noqa: E402


def row(y, xs, w=40, h=80):
    return [[x, y - h, x + w, y] for x in xs]


def test_plan_gap_erases_inside_one_row_and_keeps_the_rest():
    gts = np.array(row(100, range(0, 400, 50)) + row(300, range(0, 400, 50)), float)
    for seed in range(20):
        gap, kept = plan_gap(gts, np.random.default_rng(seed))
        erased = [i for i in range(len(gts)) if i not in kept]
        assert 1 <= len(erased) <= 3
        assert all(covered(gts[i], gap) > 0.9 for i in erased)                     # erased products are inside the hole
        assert gts[erased, 0].min() > 0 and gts[erased, 2].max() < 390            # never the first or last product
        assert len({int(gts[i, 3]) for i in erased}) == 1                           # all from one row


def test_plan_gap_skips_rows_too_short_or_too_crowded():
    assert plan_gap(np.array(row(100, [0, 50, 100, 150]), float), np.random.default_rng(0)) is None  # 4 products < 5
    # Neighbours overlapping the middle products by 30 px leave no room to erase them completely.
    crowded = np.array(row(100, [0, 10, 20, 30, 40, 50]), float)
    assert plan_gap(crowded, np.random.default_rng(0)) is None


def test_gap_mask_marks_only_the_hole():
    m = gap_mask((100, 200, 3), np.array([10.2, 20.0, 50.0, 60.0]))
    assert m.shape == (100, 200) and m[20:60, 10:50].min() == 255 and m.sum() == 255 * 40 * 40


def test_match_gaps():
    true_gap = np.array([100, 0, 200, 50])
    assert match_gaps(np.array([[110, 5, 190, 45, 0]]), true_gap) == (True, 0)
    assert match_gaps(np.array([[110, 60, 190, 90, 0]]), true_gap) == (False, 1)                      # other shelf row
    assert match_gaps(np.array([[0, 5, 400, 45, 0], [500, 5, 600, 45, 1]]), true_gap) == (False, 2)    # far too wide
    assert match_gaps(np.zeros((0, 5)), true_gap) == (False, 0)


class FakeLamaSession:
    def __init__(self, value):
        self.value = value

    def run(self, _, feed):
        assert feed["image"].shape == (1, 3, 512, 512) and feed["mask"].shape == (1, 1, 512, 512)
        assert feed["image"].max() <= 1 and set(np.unique(feed["mask"])) <= {0.0, 1.0}
        return [np.full((1, 3, 512, 512), self.value, np.float32)]


def lama(value):
    m = object.__new__(LamaInpainter)
    m.sess = FakeLamaSession(value)
    return m


def test_lama_changes_only_the_hole_and_handles_both_output_scales():
    bgr = np.random.default_rng(0).integers(0, 255, (600, 900, 3), dtype=np.uint8)
    mask = gap_mask(bgr.shape, np.array([400, 250, 480, 350]))
    for value, expected in [(128.0, 128), (0.5, 127)]:  # model returns 0-255, or 0-1 from another export
        out = lama(value)(bgr, mask)
        assert (out[mask == 0] == bgr[mask == 0]).all()
        assert (np.abs(out[mask > 0].astype(int) - expected) <= 1).all()


def test_lama_with_empty_mask_returns_the_photo():
    bgr = np.zeros((100, 100, 3), np.uint8)
    assert (lama(128.0)(bgr, np.zeros((100, 100), np.uint8)) == bgr).all()
