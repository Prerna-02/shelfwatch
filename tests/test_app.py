"""Headless tests of the Streamlit pages (no browser): they render, and a real shelf photo gives a full result.
Needs models/shelf_yolo11n_960.onnx and models/coco_yolo11n.onnx; skipped when they are missing."""
from pathlib import Path

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
needs_models = pytest.mark.skipif(not (ROOT / "models" / "shelf_yolo11n_960.onnx").exists()
                                  or not (ROOT / "models" / "coco_yolo11n.onnx").exists(), reason="models not downloaded")

# Runs the real app, but hands it a photo as if the user had uploaded one (AppTest cannot click an upload button).
WITH_PHOTO = """
import sys, streamlit as st
sys.path.insert(0, APP_DIR)
class Upload:
    def getvalue(self): return open(PHOTO, "rb").read()
st.file_uploader = lambda *a, **k: Upload()
exec(open(APP, encoding="utf8").read(), {"__file__": APP, "__name__": "__main__"})
"""


@needs_models
def test_page_waits_for_a_photo():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert "How to use" in at.sidebar.header[0].value                                 # the short help, in the sidebar
    assert "must not be used to monitor or judge staff" in " ".join(m.value for m in at.sidebar.markdown)


@needs_models
def test_page_shows_products_gaps_and_restock_list(tmp_path):
    photo = tmp_path / "shelf.jpg"
    rng = np.random.default_rng(0)
    cv2.imwrite(str(photo), rng.integers(0, 255, (900, 1200, 3), dtype=np.uint8))  # replaced below if a real photo exists
    real = sorted((ROOT / "data" / "own_shelves").glob("shelf_05.*"))
    script = tmp_path / "run_app.py"
    script.write_text(f"APP_DIR = {str(ROOT / 'app')!r}\nAPP = {str(APP)!r}\nPHOTO = {str(real[0] if real else photo)!r}\n"
                      + WITH_PHOTO, encoding="utf8")
    at = AppTest.from_file(str(script), default_timeout=120).run()
    assert not at.exception
    banner = (at.error or at.success)[0].value
    assert "to restock" in banner and "% full" in banner
    assert at.toggle[0].label == "Show products" and at.toggle[0].value is False      # products hidden by default
    if real:  # our own dairy-shelf photo: has real empty shelves and two shoppers
        assert banner.startswith("**") and "gap" in banner and at.subheader[0].value == "Restock first"
        assert any("behind people" in w.value for w in at.warning)
        assert len(at.get("progress")) > 0                                             # one bar per shelf row

