"""Object detection and spatial metadata generation with graceful error handling."""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

YOLO: Any = None

MODEL_PATH = "yolo11m.pt"
ANNOTATED_FILENAME = "output_annotated.jpg"

# Image extensions we consider valid before sending to YOLO.
_VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".tif", ".tiff", ".webp", ".avif",
}
_YOLO_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".avif",
}


def _as_list(value: Any) -> list[Any]:
    """Convert tensors and array-like values to regular Python lists."""
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        return [value]
    return value


def _spatial_location(
    bbox: list[float], image_width: float, image_height: float
) -> str:
    """Return the cell containing the bounding box center in a 3x3 grid."""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    column = min(2, max(0, int((center_x / image_width) * 3)))
    row = min(2, max(0, int((center_y / image_height) * 3)))
    column_name = ("left", "center", "right")[column]
    row_name = ("top", "center", "bottom")[row]
    return f"{row_name}-{column_name}"


def _image_dimensions(result: Any, image_path: Path) -> tuple[float, float]:
    """Get image dimensions from a YOLO result, falling back to OpenCV."""
    original_shape = getattr(result, "orig_shape", None)
    if original_shape is not None and len(original_shape) >= 2:
        height, width = original_shape[:2]
        if width and height:
            return float(width), float(height)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image file: {image_path}")
    height, width = image.shape[:2]
    if not width or not height:
        raise ValueError(f"Image has invalid dimensions: {image_path}")
    return float(width), float(height)


def _save_verified_annotated_image(
    source_path: Path,
    detections: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Draw only the deduplicated detections shown in the evidence ledger."""
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image for annotation: {source_path}")

    height, width = image.shape[:2]
    grid = image.copy()
    for column in (1, 2):
        x = round(width * column / 3)
        cv2.line(grid, (x, 0), (x, height), (120, 120, 120), 1, cv2.LINE_AA)
    for row in (1, 2):
        y = round(height * row / 3)
        cv2.line(grid, (0, y), (width, y), (120, 120, 120), 1, cv2.LINE_AA)
    image = cv2.addWeighted(grid, 0.3, image, 0.7, 0)

    for item in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in item["bbox"]]
        label = f'{item["class"]} {float(item["confidence"]) * 100:.0f}%'
        color = (0, 215, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        )
        text_top = max(0, y1 - text_height - baseline - 6)
        cv2.rectangle(
            image,
            (x1, text_top),
            (x1 + text_width + 10, y1),
            color,
            thickness=-1,
        )
        cv2.putText(
            image,
            label,
            (x1 + 5, max(text_height + 2, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Unable to save annotated image: {output_path}")


def _validate_image_file(source_path: Path) -> None:
    """Check that the file exists, is a file, is non-empty, and readable."""
    if not source_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {source_path}")
    if not source_path.is_file():
        raise ValueError(f"Image path is not a file: {source_path}")
    if source_path.stat().st_size == 0:
        raise ValueError(f"Image file is empty (0 bytes): {source_path}")
    ext = source_path.suffix.lower()
    if ext and ext not in _VALID_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{ext}'. "
            f"Supported: {', '.join(sorted(_VALID_EXTENSIONS))}"
        )
    # Quick readability check via OpenCV header
    test = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if test is None:
        raise ValueError(
            f"Image file appears corrupted or unreadable: {source_path}"
        )


def _prepare_yolo_source(source_path: Path, temporary_directory: Path) -> Path:
    """Return a path with a suffix supported by Ultralytics."""
    if source_path.suffix.lower() in _YOLO_EXTENSIONS:
        return source_path

    staged_path = temporary_directory / "input.jpg"
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None or not cv2.imwrite(str(staged_path), image):
        raise ValueError(
            f"Unable to convert image to a YOLO-compatible format: {source_path}"
        )
    return staged_path


def _intersection_over_union(first: list[float], second: list[float]) -> float:
    """Return overlap between two bounding boxes."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection == 0:
        return 0.0
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _overlap_over_smaller_box(first: list[float], second: list[float]) -> float:
    """Return overlap as a fraction of the smaller box."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    smaller_area = min(first_area, second_area)
    return intersection / smaller_area if smaller_area else 0.0


def detect_objects(image_path: str, conf_threshold: float = 0.35) -> str:
    """Detect objects and return JSON metadata for an image.

    The detector writes ``output_annotated.jpg`` in the current working
    directory. The model is loaded lazily so importing this module never
    downloads weights.
    """
    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError("image_path must be a non-empty string")
    if not isinstance(conf_threshold, (int, float)) or not 0.1 <= conf_threshold <= 0.9:
        raise ValueError("conf_threshold must be between 0.1 and 0.9")

    source_path = Path(image_path).expanduser()

    # Validate image before YOLO
    _validate_image_file(source_path)

    # Load YOLO model
    yolo_class = YOLO
    if yolo_class is None:
        try:
            from ultralytics import YOLO as yolo_class
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "Ultralytics could not be loaded. Install its runtime dependencies."
            ) from error

    with tempfile.TemporaryDirectory(prefix="vision-agent-") as temporary_directory:
        inference_path = _prepare_yolo_source(source_path, Path(temporary_directory))
        try:
            model = yolo_class(MODEL_PATH)
            results = list(model(str(inference_path), conf=float(conf_threshold)))
        except Exception as error:
            raise RuntimeError(
                f"YOLO inference failed: {error}. "
                "The image may be corrupted or an unsupported format."
            ) from error
        if not results:
            raise ValueError("YOLO returned no results")
        result = results[0]
        output_path = Path.cwd() / ANNOTATED_FILENAME

    image_width, image_height = _image_dimensions(result, source_path)

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        raise ValueError("YOLO result did not contain boxes")

    coordinates = _as_list(getattr(boxes, "xyxy", None))
    confidences = _as_list(getattr(boxes, "conf", None))
    class_ids = _as_list(getattr(boxes, "cls", None))
    names = getattr(result, "names", {}) or {}
    candidates: list[dict[str, Any]] = []

    for index, coordinates_item in enumerate(coordinates):
        if len(coordinates_item) < 4:
            raise ValueError("YOLO returned an invalid bounding box")
        bbox = [float(value) for value in coordinates_item[:4]]
        confidence = float(confidences[index]) if index < len(confidences) else 0.0
        class_id = int(class_ids[index]) if index < len(class_ids) else 0
        if isinstance(names, dict):
            class_name = str(names.get(class_id, class_id))
        elif 0 <= class_id < len(names):
            class_name = str(names[class_id])
        else:
            class_name = str(class_id)

        candidates.append(
            {
                "class": class_name,
                "bbox": bbox,
                "confidence": confidence,
                "spatial_location": _spatial_location(
                    bbox, image_width, image_height
                ),
            }
        )

    # Guard against duplicate same-class boxes that can occasionally survive
    # model post-processing for a single visible object.
    detections: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates, key=lambda item: item["confidence"], reverse=True
    ):
        duplicate = any(
            candidate["class"] == existing["class"]
            and (
                _intersection_over_union(candidate["bbox"], existing["bbox"]) >= 0.7
                or _overlap_over_smaller_box(
                    candidate["bbox"], existing["bbox"]
                )
                >= 0.85
            )
            for existing in detections
        )
        if not duplicate:
            detections.append(candidate)

    _save_verified_annotated_image(source_path, detections, output_path)

    metadata = {
        "counts": dict(Counter(item["class"] for item in detections)),
        "detections": detections,
        "annotated_image_path": str(output_path.resolve()),
    }
    return json.dumps(metadata, indent=2)
