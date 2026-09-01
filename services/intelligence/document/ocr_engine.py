"""Tesseract OCR engine — image to text extraction."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np


class OCREngine:
    """OCR using Tesseract."""

    def __init__(self):
        # Check common Tesseract install paths
        import os
        tess_paths = [
            r"C:\Program Files\Tesseract-OCR",
            r"/usr/bin",
            r"/usr/local/bin",
            r"/opt/homebrew/bin",
        ]
        tess_dir = None
        for p in tess_paths:
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, "tesseract.exe")):
                tess_dir = p
                break
            if shutil.which("tesseract"):
                tess_dir = None  # already in PATH
                break
        if tess_dir:
            os.environ["PATH"] = tess_dir + ";" + os.environ.get("PATH", "")
        self._available = shutil.which("tesseract") is not None
        self._pytesseract = None
        if self._available:
            try:
                import pytesseract
                self._pytesseract = pytesseract
            except ImportError:
                self._available = False

    def available(self) -> bool:
        return self._available

    def ocr_image(self, image_path: str, lang: str = "eng") -> str:
        """Extract text from an image file."""
        if not self._available:
            raise RuntimeError("Tesseract not installed")
        from PIL import Image
        img = Image.open(image_path)
        return self._pytesseract.image_to_string(img, lang=lang)

    def ocr_array(self, img: np.ndarray, lang: str = "eng") -> str:
        """Extract text from a numpy array."""
        if not self._available:
            raise RuntimeError("Tesseract not installed")
        from PIL import Image
        if len(img.shape) == 3 and img.shape[2] == 3:
            import cv2
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        return self._pytesseract.image_to_string(pil_img, lang=lang)

    def ocr_data(self, image_path: str, lang: str = "eng") -> dict:
        """Extract structured OCR data with bounding boxes."""
        if not self._available:
            raise RuntimeError("Tesseract not installed")
        from PIL import Image
        img = Image.open(image_path)
        return self._pytesseract.image_to_data(img, lang=lang, output_type=self._pytesseract.Output.DICT)
