# ShelfWatch: shelf-gap and out-of-stock detector

MAIB AI 221 Computer Vision, Final Project 4 (Retail).

Upload one photo of a store shelf. ShelfWatch finds every product, highlights the
empty gaps, shows how full each shelf row is, and gives a ranked restock list as a CSV.
People in the photo are blurred, and no image is stored.

**Live app:** https://shelfwatch.streamlit.app/

Architecture diagrams: [docs/architecture.md](docs/architecture.md).

## Project structure

```
shelfwatch/
├── app/                     # Everything that runs on Streamlit Community Cloud
│   ├── streamlit_app.py     # Entry point: upload, results, CSV download
│   ├── pipeline.py          # Quality check, blur, detect, rows, gaps (no Streamlit code)
│   (the short help required by the brief is a collapsed section on the main screen)
├── models/
│   ├── shelf_yolo11n_960.onnx   # Fine-tuned shelf-product detector (from the notebook, 10 MB)
│   └── coco_yolo11n.onnx        # Off-the-shelf COCO YOLO11n, used only to find and blur people (11 MB)
├── notebooks/
│   ├── shelfwatch.ipynb     # The project notebook (Colab): data, before/after, evaluation, generative AI, own photos
│   └── shelf_utils.py       # Its helpers: dataset stats, scoring, drawing, synthetic gaps (LaMa / classical)
├── tools/prelabel.py        # Pre-draws product boxes on new photos so labelling is correcting, not drawing
├── tests/                   # pytest: pipeline, synthetic-gap helpers, and the Streamlit pages (headless)
├── docs/                    # Architecture, labelling guide (gap rule), model card
├── data/                    # Local only, git-ignored (see "Data" below)
├── requirements.txt         # App runtime dependencies only
├── requirements-dev.txt     # Plus pandas, matplotlib, pillow, pytest for local tests
└── .gitignore
```

Training libraries (`ultralytics`, `torch`, LaMa/MI-GAN) are installed inside the
Colab notebook, not in `requirements.txt`. That keeps the deployed app small enough
for Streamlit Community Cloud's free memory limit.

`shelfwatch.ipynb` runs on Colab (GPU about 1.5 h, CPU about 2.5 h). Before running it, copy `app/pipeline.py` and
`notebooks/shelf_utils.py` to Google Drive at `MyDrive/shelfwatch/code/` (replacing older copies), so the notebook
tests the same code the app runs. Part 5 reads the 40 real shelf photos from `MyDrive/shelfwatch/own_shelves/`.

## Data

`data/` is git-ignored: the datasets are large, SKU-110K is research-use only, and the
own-shelf photos were taken with store permission and are not ours to publish.

```
data/
├── sku110k_subset/   # Not used locally: SKU-110K is downloaded on Colab by the notebook
├── own_shelves/      # 40 local shelf photos: prelabels/ (model boxes), labels/ (hand-corrected, with gaps)
└── synthetic/        # Inpainted empty-shelf images with known gap positions
```

- SKU-110K: Goldman et al., "Precise Detection in Densely Packed Scenes", CVPR 2019.
  Research use only; this app is not for commercial use.
- Ultralytics YOLO is AGPL-3.0, so this repository is public under AGPL-3.0
  (add the licence with GitHub's "Add file → Create new file → LICENSE" picker).

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate on macOS/Linux
pip install -r requirements-dev.txt
python -m pytest tests
streamlit run app/streamlit_app.py
```

## Deploy

Streamlit Community Cloud → Create app → this repository, branch `main`, main file `app/streamlit_app.py`, and under Advanced settings choose Python 3.11 (the version the tests run on). Every push to `main` updates the live app.
