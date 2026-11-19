"""ShelfWatch entry point for Streamlit Community Cloud: one shelf photo in, gaps to restock out.
UI only: all image logic lives in pipeline.py. Nothing is saved; the photo exists only while the page handles it."""
import csv
import io
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from pipeline import ProductDetector, analyse, draw_result, gap_table

MODELS = Path(__file__).resolve().parents[1] / "models"
MAX_SIDE = 2000  # phone photos are often 4000 px; shrinking keeps memory and time low without losing products

st.set_page_config(page_title="ShelfWatch", page_icon=":material/storefront:", layout="wide")


@st.cache_resource
def load_models():
    """Loaded once per server, shared by all visitors."""
    return (ProductDetector(str(MODELS / "shelf_yolo11n_960.onnx")),
            ProductDetector(str(MODELS / "coco_yolo11n.onnx"), size=640))


def side(g, width):
    x = (g[0] + g[2]) / 2 / width
    return "left side" if x < 0.34 else "right side" if x > 0.66 else "middle"


def gap_card(image, g, number, width):
    """One restock card: a close-up of the gap and where it is, in words."""
    x1, y1, x2, y2 = g[:4].astype(int)
    pad = int(0.6 * max(x2 - x1, y2 - y1))
    crop = cv2.rectangle(image.copy(), (x1, y1), (x2, y2), (0, 0, 255), max(2, image.shape[1] // 300))
    crop = crop[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
    with st.container(border=True):
        pic, text = st.columns([1, 2], vertical_alignment="center")
        pic.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), width="stretch")
        text.markdown(f"**{number} · Shelf {int(g[4]) + 1} from top**  \n{side(g, width).capitalize()} · "
                      f"about {max(1, round(float(g[5])))} product{'s' if round(float(g[5])) > 1 else ''} wide")


def how_to():
    with st.sidebar:
        st.header(":material/help: How to use", anchor=False)
        st.markdown("""
- Stand in front of **one shelf unit** and hold the phone **straight**, not at an angle along the aisle.
- Fit the whole unit in the frame; avoid glare from lights and glass fridge doors; hold still.
- **Red** gaps look empty: restock them. **Orange** spots look busy: often a product the app missed, so check by eye.
- The app finds **empty space**, not **which** product is missing, and it is weakest on flat items lying down,
  thin hanging items and glass fridge doors. A person checks the shelf before restocking.
- **Privacy:** people are pixelated automatically and **no photo is stored**. ShelfWatch is for restocking shelves and
  **must not be used to monitor or judge staff**. Research and teaching project (MAIB AI 221); not for commercial use.
""")


how_to()
st.title(":material/storefront: ShelfWatch", anchor=False)
st.caption("Find empty shelf space from one photo")

source = st.segmented_control("Photo from", ["Upload", "Camera"], default="Upload", label_visibility="collapsed")
photo = (st.camera_input("Take a photo of the shelf", label_visibility="collapsed") if source == "Camera" else
         st.file_uploader("Upload a shelf photo", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed"))
if photo is None:
    st.stop()

bgr = cv2.imdecode(np.frombuffer(photo.getvalue(), np.uint8), cv2.IMREAD_COLOR)
if bgr is None:
    st.error("This file isn't a photo the app can read. Upload a JPG or PNG.")
    st.stop()
if max(bgr.shape[:2]) > MAX_SIDE:
    s = MAX_SIDE / max(bgr.shape[:2])
    bgr = cv2.resize(bgr, (round(bgr.shape[1] * s), round(bgr.shape[0] * s)), interpolation=cv2.INTER_AREA)

with st.spinner("Looking for gaps..."):
    t0 = time.time()
    r = analyse(bgr, *load_models())
    print(f"shelfwatch: processed {bgr.shape[1]}x{bgr.shape[0]} in {time.time() - t0:.2f} s")  # speed, in the server log

W = bgr.shape[1]
restock = [g for g, e in zip(r["gaps"], r["empty"]) if e]
check = [g for g, e in zip(r["gaps"], r["empty"]) if not e]
full = round(100 * np.mean(r["fullness"])) if r["fullness"] else 0

banner, download = st.columns([4, 1], vertical_alignment="center")
summary = f"Shelf is {full}% full · {len(check)} spot{'s' if len(check) != 1 else ''} to double-check"
if restock:
    banner.error(f"**{len(restock)} gap{'s' if len(restock) != 1 else ''} to restock**  \n{summary}",
                 icon=":material/inventory_2:")
else:
    banner.success(f"**No gaps to restock**  \n{summary}", icon=":material/check_circle:")
table = gap_table(r["gaps"], W, r["empty"])
if table:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(table[0]))
    writer.writeheader(); writer.writerows(table)
    download.download_button("Restock list", buf.getvalue(), "restock_list.csv", "text/csv",
                             icon=":material/download:", width="stretch")
for p in r["problems"]:
    st.warning(p, icon=":material/warning:")

left, right = st.columns([3, 2], gap="large")
with left:
    show_products = st.toggle("Show products", value=False)
    st.image(cv2.cvtColor(draw_result(r["image"], r["boxes"] if show_products else np.zeros((0, 4)),
                                      r["gaps"], r["empty"]), cv2.COLOR_BGR2RGB), width="stretch")
    st.markdown(":red[**■**] Restock &nbsp;&nbsp; :orange[**■**] Double-check"
                + (" &nbsp;&nbsp; :green[**■**] Product" if show_products else ""))

with right:
    if restock:
        st.subheader("Restock first")
        for i, g in enumerate(restock, 1):
            gap_card(r["image"], g, i, W)
    if check:
        with st.expander(f"Double-check {len(check)} spot{'s' if len(check) != 1 else ''}", icon=":material/visibility:"):
            for i, g in enumerate(check, len(restock) + 1):
                gap_card(r["image"], g, i, W)
    if r["fullness"]:
        st.subheader("Shelf rows")
        for i, f in enumerate(r["fullness"], 1):
            st.progress(min(max(f, 0.0), 1.0), text=f"Shelf {i} from top · {round(100 * f)}% full")
