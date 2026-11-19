# ShelfWatch architecture

Two halves: what is built once on Colab, and what the app does for every photo.

## 1. Build (offline, on Colab)

```mermaid
flowchart TD
    A[SKU-110K subset] --> B[Fine-tune YOLO11n]
    A --> C[Inpaint products away<br/>LaMa or MI-GAN]
    C --> D[Synthetic empty shelves<br/>gaps at known positions]
    D -.extra training data.-> B
    B --> E[Export to ONNX]
    E --> F[Evaluate on held-out data]
    D --> F
    G[Own 40 shelf photos<br/>test only, never trained on] --> F
    F --> H[models/ folder]
```

## 2. App (every uploaded photo, on Streamlit Community Cloud)

```mermaid
flowchart TD
    P[Shelf photo] --> Q{Quality check<br/>blur, size, brightness}
    Q -- fails --> N[Not sure: ask to retake]
    Q -- passes --> B[Blur people<br/>COCO YOLO11n person class]
    B --> D{Detect products<br/>shelf YOLO11n ONNX + NMS}
    D -- low confidence --> N
    D --> R[Group boxes into shelf rows]
    R --> G[Find gaps, fullness % per row]
    G --> S[Show overlay, ranked gaps, CSV download]
```

Nothing is saved: the photo lives in memory for one request only.
