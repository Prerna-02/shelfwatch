# Labelling guide: own shelf photos

Fixed **before** looking at any gap results, so the test cannot be tuned to the app.

## Two classes

| Class | Draw a box around |
|---|---|
| `product` | Every product whose **front faces the camera**: one box per item, tight to its visible edges. |
| `gap` | Every **empty shelf space at least half a product wide** (half the width of the products next to it in that row). |

## Gap rule

- **Width:** from the edge of the last product on the left to the first product on the right. Neighbouring empty spaces with no product between them are **one** gap.
- **Height:** from the shelf surface up to the height of the neighbouring products in that row.
- **Shelf ends:** empty space between the last product and the end of the shelf counts as a gap too.
- **Not a gap:** a product pushed to the back but still visible; price tags, shelf edges, signs; space above the top shelf; space outside the shelving unit.
- **Unsure?** Leave it out and add the photo name to the notes column of `data/own_shelves/photo_list.csv`.

## Steps (makesense.ai, free, no account)

1. Open makesense.ai → *Get Started* → drop in all photos from `data/own_shelves/` → *Object Detection*.
2. Load the class names from `data/own_shelves/prelabels/labels.txt` (product, gap).
3. *Actions → Import Annotations → YOLO*: select all `prelabels/shelf_*.txt` plus `labels.txt`. These are the model's product boxes.
4. Per photo: delete wrong product boxes, add missed products, then draw every gap box yourself.
5. *Actions → Export Annotations → YOLO* often (the tool does not keep your work if the tab closes).
   Unzip the export into `data/own_shelves/labels/` (a separate folder: re-running `tools/prelabel.py` never touches it).

Pre-labels come from the model at confidence 0.35 (the app's threshold). Gaps are never pre-labelled: the gap finder is what this test set checks.
