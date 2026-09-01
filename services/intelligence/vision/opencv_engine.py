"""OpenCV vision engine — image preprocessing, video frame extraction, transforms."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class OpenCVEngine:
    """Image and video processing using OpenCV."""

    def load_image(self, path_or_bytes: str | bytes) -> np.ndarray:
        """Load image from file path or bytes."""
        if isinstance(path_or_bytes, bytes):
            arr = np.frombuffer(path_or_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(str(path_or_bytes))
        if img is None:
            raise ValueError(f"Failed to load image: {path_or_bytes}")
        return img

    def save_image(self, img: np.ndarray, path: str, quality: int = 95) -> str:
        """Save image to file."""
        ext = Path(path).suffix or ".jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
        cv2.imwrite(str(path), img, params)
        return path

    def resize(self, img: np.ndarray, width: int = 0, height: int = 0,
               max_dim: int = 0) -> np.ndarray:
        """Resize image, preserving aspect ratio."""
        h, w = img.shape[:2]
        if max_dim:
            scale = max_dim / max(h, w)
            if scale < 1:
                width, height = int(w * scale), int(h * scale)
        if width and not height:
            height = int(h * width / w)
        elif height and not width:
            width = int(w * height / h)
        if width and height:
            return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        return img

    def to_grayscale(self, img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def to_rgb(self, img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def detect_edges(self, img: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
        gray = self.to_grayscale(img) if len(img.shape) == 3 else img
        return cv2.Canny(gray, low, high)

    def extract_frames(self, video_path: str, max_frames: int = 10,
                       interval: int = 30) -> list[np.ndarray]:
        """Extract frames from video at regular intervals."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // max_frames) if max_frames else interval

        idx = 0
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frames.append(frame)
            idx += 1
        cap.release()
        return frames

    def get_video_info(self, video_path: str) -> dict:
        """Get video metadata."""
        cap = cv2.VideoCapture(str(video_path))
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
        }
        cap.release()
        if info["fps"] > 0:
            info["duration_s"] = info["frames"] / info["fps"]
        return info

    @staticmethod
    def available() -> bool:
        return True  # opencv-python is always available after pip install
