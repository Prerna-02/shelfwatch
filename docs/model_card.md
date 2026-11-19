# Model card: ShelfWatch shelf-gap detector

**What it is.** A YOLO11n object detector fine-tuned to find products on supermarket shelves (one class: product),
exported to ONNX (10.4 MB) and run on CPU, plus rules that turn product boxes into gaps: shelf rows from box positions,
empty stretches of at least half a product, and an edge-based check that a gap really looks empty.
Live app: https://shelfwatch.streamlit.app · Code: https://github.com/Prerna-02/shelfwatch

## Intended use
- A store associate photographs one shelf unit with a phone and gets a ranked restock list (where, how wide).
- Decision support only: a person looks at the shelf before restocking. Red = looks empty, orange = check by eye.

## Out-of-scope use
- Monitoring, timing or judging staff, or identifying shoppers (people are pixelated; nothing is stored).
- Deciding *which* product is missing (the model has one class and reads no labels), automatic ordering, or any
  commercial use (SKU-110K is licensed for research only).

## Data
- **Training:** SKU-110K (Goldman et al., CVPR 2019, research use), 30% subset = 2,466 photos, 50 epochs, 960 px,
  COCO-pretrained YOLO11n (Ultralytics, AGPL-3.0). Validation 588 photos; test 2,935 photos (1 corrupt file skipped).
- **Real-world test:** 40 photos taken by us in local stores with permission; 108 gaps drawn by hand following
  `docs/labelling_guide.md`, before looking at the app's gaps. Not published.

## Metrics
| Test | Result |
|---|---|
| Products, SKU-110K test (2,935 photos) | mAP@0.5 **0.92** (off-the-shelf COCO model: 0.30), precision 0.90, recall 0.85 |
| Product count error per photo | **10%** (COCO model: 97%) |
| Gaps, own 40 photos, red "restock" list | recall **0.67**, precision **0.54**, F1 **0.60** |
| Gaps, own photos, red + orange | recall **0.81** |
| Emptiness check (tuned on 20 photos, tested on 20 others) | precision 0.29 → 0.55, recall 0.86 → 0.65 |
| Speed (CPU) | about 0.5 s per photo on a laptop; brief limit 5 s |

Confidence threshold 0.35, chosen on validation photos (best F1).

## Subgroup results
| Group | Result |
|---|---|
| SKU-110K, by shelf density (thirds of 300 test photos) | mAP@0.5: sparse 0.91, medium 0.94, **dense 0.91**; recall falls to **0.82** on the densest shelves |
| Own photos, open shelves (27 photos, 93 gaps) | gap F1 **0.65** (recall 0.68, precision 0.62) |
| Own photos, glass-door fridges (5 photos, 12 gaps) | gap F1 **0.61** (recall 0.67, precision 0.56) |
| Own photos, chocolate / flat bars (8 photos, 3 gaps) | gap F1 **0.09**: flat bars are missed, so their space looks empty |

Small groups (5 and 8 photos) give rough numbers only.

## Known failures
- **Flat items lying down** (chocolate bars) and **thin hanging items** (toothbrushes on pegs): missed products become
  false gaps.
- **Glass fridge doors:** the metal frame between doors can look like a gap; reflections and lights confuse it.
- **Completely empty shelves and empty shelf ends:** shelves are found from the products on them, so a shelf with no
  products, or an empty end no neighbouring product reaches, can be missed.
- **Long aisles photographed at an angle:** far products are tiny and missed.
- **Generative test data:** LaMa inpainting repaints products into erased holes (detector sees a product in 83% of
  fills), so it could not create realistic empty shelves; a classical fill was more useful for testing.

## Responsible AI
- **Privacy:** people are detected with an off-the-shelf COCO YOLO11n and pixelated before the photo is shown; no
  photo is stored or logged; the public notebook omits our store photos.
- **Human in the loop:** red and orange gaps, "not sure" warnings (blurry, dark, too few products, people hiding the
  shelf, low confidence), and the instruction to check the shelf before acting.
- **Fairness and drift:** trained on SKU-110K stores, mostly outside our region; accuracy drops on local formats
  (chocolate displays, fridges). Re-test when moving to a new store layout or country.
