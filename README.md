# Vision Agent

Vision Agent combines Ultralytics YOLO object detection with a Groq
tool-calling assistant and a Gradio UI.

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

   ```text
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set `GROQ_API_KEY`.
4. Start the UI:

   ```text
   python app.py
   ```

The first detection downloads `yolo11m.pt`. The annotated output is written
to `output_annotated.jpg` in the current working directory. The UI includes a
confidence threshold slider, a 3x3 spatial grid, and follow-up agent chat
grounded in cached detector metadata.

The UI presents the full observable pipeline after each run: frame ingest,
YOLO detection, evidence verification, and grounded Groq synthesis. It also
shows the annotated frame plus a detection inventory with confidence and
3x3 spatial position for every verified object.

## Offline tests

Tests mock both YOLO and Groq, so they do not download model weights or call
the live API:

```text
python -m pytest tests -q
```

## Detection metadata

`tools.vision_tool.detect_objects(image_path, conf_threshold=0.35)` returns JSON
containing object counts, bounding boxes, confidence values, 3x3 spatial
labels, and the annotated image path.
