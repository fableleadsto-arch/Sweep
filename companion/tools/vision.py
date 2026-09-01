"""Computer-vision capability — OpenCV + Pillow (+ optional local vision model).

Image preprocessing and feature extraction run on OpenCV/Pillow; when
`params.use_model` is set and Transformers/PyTorch are installed, a small
local classification model is run on top (OpenCV preprocessing → vision model
→ answer). All imports are lazy.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Optional

DEFAULT_VISION_MODEL = "google/mobilenet_v2_1.0_224"


def run_vision(payload: dict[str, Any]) -> dict[str, Any]:
    """Analyze an image (decode → preprocess → describe/features/model)."""
    params = payload.get("params") or {}
    operation = str(params.get("operation") or "describe").lower()

    pil_image = _load_image(payload)
    from PIL import Image

    libraries_used = ["pillow"]

    np = _load("numpy")
    cv2 = _load("cv2")

    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    libraries_used.append("opencv")

    if operation == "resize":
        width = int(params.get("width") or 640)
        height = int(params.get("height") or int(width * pil_image.height / max(1, pil_image.width)))
        resized = cv2.resize(bgr, (width, height))
        out = _to_base64(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        return {
            "result": {"operation": "resize", "size": [width, height], "image_base64": out},
            "summary": f"Resized image to {width}×{height}.",
            "libraries_used": libraries_used,
        }

    if operation == "grayscale":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        out = _to_base64(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
        return {
            "result": {"operation": "grayscale", "image_base64": out},
            "summary": "Converted image to grayscale.",
            "libraries_used": libraries_used,
        }

    if operation == "rotate":
        degrees = float(params.get("degrees") or 90)
        rows, cols = bgr.shape[:2]
        matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), degrees, 1.0)
        rotated = cv2.warpAffine(bgr, matrix, (cols, rows))
        out = _to_base64(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
        return {
            "result": {"operation": "rotate", "degrees": degrees, "image_base64": out},
            "summary": f"Rotated image {degrees:g}°.",
            "libraries_used": libraries_used,
        }

    if operation == "blur":
        kernel = int(params.get("kernel") or 5)
        blurred = cv2.GaussianBlur(bgr, (kernel, kernel), 0)
        out = _to_base64(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
        return {
            "result": {"operation": "blur", "kernel": kernel, "image_base64": out},
            "summary": f"Gaussian-blurred image (kernel {kernel}).",
            "libraries_used": libraries_used,
        }

    if operation == "threshold":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        threshold = int(params.get("threshold") or 127)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        out = _to_base64(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))
        return {
            "result": {"operation": "threshold", "threshold": threshold, "image_base64": out},
            "summary": f"Binary threshold at {threshold}.",
            "libraries_used": libraries_used,
        }

    if operation == "features":
        orb = cv2.ORB_create()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        return {
            "result": {
                "operation": "features",
                "keypoints": int(len(keypoints)),
                "descriptor_size": None if descriptors is None else descriptors.shape,
            },
            "summary": f"Detected {len(keypoints)} ORB feature keypoints.",
            "libraries_used": libraries_used,
        }

    if operation == "faces":
        face_count, boxes = _detect_faces(cv2, bgr)
        return {
            "result": {"operation": "faces", "faces": face_count, "boxes": boxes},
            "summary": f"Detected {face_count} face(s).",
            "libraries_used": libraries_used,
        }

    # Default: descriptive profile.
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    avg_color = tuple(float(v) for v in np.mean(gray, axis=(0, 1)))
    gray_luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray_luma))
    edges = cv2.Canny(gray_luma, 100, 200)
    edge_ratio = float(np.count_nonzero(edges) / max(1, edges.size))
    desc = {
        "operation": "describe",
        "width": int(pil_image.width),
        "height": int(pil_image.height),
        "mode": str(pil_image.mode),
        "format": str(pil_image.format or "unknown"),
        "aspect_ratio": round(pil_image.width / max(1, pil_image.height), 4),
        "average_color_rgb": [round(v, 2) for v in avg_color],
        "brightness": round(brightness, 2),
        "edge_density": round(edge_ratio, 4),
    }
    summary = (
        f"Image {pil_image.width}×{pil_image.height} ({pil_image.mode}), average RGB "
        f"{tuple(round(v, 1) for v in avg_color)}, brightness {brightness:.0f}/255."
    )

    # Optional local vision model on top (OpenCV preprocessing → model → result).
    if params.get("use_model"):
        model_result = _run_local_vision_model(payload, pil_image)
        if model_result is not None:
            desc["model"] = model_result
            summary += f" Local model labels: {model_result.get('top_label')} ({model_result.get('top_score')})"

    return {"result": desc, "summary": summary, "libraries_used": libraries_used}


def _load_image(payload: dict[str, Any]) -> Any:
    raw = payload.get("image_base64")
    if raw is None:
        data = payload.get("data")
        if isinstance(data, str):
            raw = data
        elif isinstance(data, dict) and isinstance(data.get("image_base64"), str):
            raw = data["image_base64"]
    if not raw:
        raise ValueError(
            "No image provided. Send `image_base64` as a data URL or raw base64 string."
        )
    if raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    image_bytes = base64.b64decode(raw)
    from PIL import Image

    return Image.open(io.BytesIO(image_bytes))


def _to_base64(rgb_array) -> str:
    from PIL import Image

    out = Image.fromarray(rgb_array)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _detect_faces(cv2, bgr) -> tuple[int, list[list[int]]]:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    try:
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return 0, []
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return int(len(faces)), [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in faces]
    except Exception:  # noqa: BLE001 - face detection is best-effort
        return 0, []


def _run_local_vision_model(payload: dict[str, Any], pil_image) -> Optional[dict[str, Any]]:
    """Zero-shot image classification with a small local model (opt-in)."""
    try:
        from transformers import pipeline  # lazy
    except Exception:  # noqa: BLE001 - model stack unavailable
        return None
    params = payload.get("params") or {}
    model_name = str(params.get("model") or DEFAULT_VISION_MODEL)
    try:
        pipe = pipeline("image-classification", model=model_name)
        predictions = pipe(pil_image, top_k=3)
        top = predictions[0] if predictions else {}
        return {
            "model": model_name,
            "top_label": top.get("label"),
            "top_score": round(float(top.get("score") or 0), 4),
            "top3": [{"label": p.get("label"), "score": round(float(p.get("score") or 0), 4)} for p in predictions],
        }
    except Exception as exc:  # noqa: BLE001 - model download/inference can fail
        return {"model": model_name, "error": str(exc)[:200]}


def _load(name: str):
    from .common import load

    return load(name)
