"""Health check — real inference verification for each installed model."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from services.intelligence.model_manager.device import detect_device
from services.intelligence.model_manager.loader import ModelLoader
from services.intelligence.model_manager.registry import ModelEntry, ModelRegistry, ModelStatus


class HealthCheck:
    """Verify each model with real inference."""

    def __init__(self):
        self.device = detect_device()
        self.registry = ModelRegistry()
        self.loader = ModelLoader(self.registry, self.device)

    def run_all(self) -> dict[str, dict]:
        """Run health checks on all registered models."""
        results = {}
        entries = self.registry.all()

        print(f"\n{'='*60}")
        print(f"  Sweep Intelligence Model Health Check")
        print(f"  Device: {self.device.device} | RAM: {self.device.ram_total_gb:.1f}GB")
        if self.device.cuda_available:
            print(f"  GPU: {self.device.gpu_name} ({self.device.gpu_memory_gb:.1f}GB)")
        print(f"{'='*60}\n")

        for i, entry in enumerate(entries, 1):
            tag = f"[{i}/{len(entries)}]"
            result = self._check_model(entry)
            results[entry.name] = result

            status_str = result["status"]
            detail = result.get("detail", "")
            if status_str == "OK":
                print(f"  {tag} [OK]        {entry.name:<30s} {detail}")
            elif status_str == "SKIP":
                print(f"  {tag} [SKIP]      {entry.name:<30s} {detail}")
            elif status_str == "UNAVAILABLE":
                print(f"  {tag} [UNAVAIL]   {entry.name:<30s} {detail}")
            else:
                print(f"  {tag} [FAIL]      {entry.name:<30s} {detail}")

        # Summary
        ok = sum(1 for r in results.values() if r["status"] == "OK")
        skip = sum(1 for r in results.values() if r["status"] == "SKIP")
        fail = sum(1 for r in results.values() if r["status"] in ("FAIL", "ERROR"))
        unavail = sum(1 for r in results.values() if r["status"] == "UNAVAILABLE")

        print(f"\n{'='*60}")
        print(f"  Results: {ok} OK | {skip} skipped | {unavail} unavailable | {fail} failed")
        print(f"{'='*60}\n")

        return results

    def _check_model(self, entry: ModelEntry) -> dict:
        """Check a single model with real inference."""
        # Models that don't need download
        if entry.provider == "opencv":
            return self._check_opencv()
        if entry.provider == "tesseract":
            return self._check_tesseract()

        # layoutlmv3 requires tesseract for image preprocessing
        if entry.name == "layoutlmv3-base" and not self._check_tesseract().get("available", False):
            import shutil
            if not shutil.which("tesseract"):
                return {"status": "UNAVAILABLE", "detail": "Requires Tesseract binary for image preprocessing"}

        if entry.status == ModelStatus.NOT_DOWNLOADED:
            return {"status": "UNAVAILABLE", "detail": "Not downloaded"}

        if entry.optional and entry.requires_gpu and self.device.device == "cpu":
            return {"status": "SKIP", "detail": "Requires GPU (optional)"}

        if entry.requires_auth:
            import os
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if not token:
                return {"status": "UNAVAILABLE", "detail": "Requires HF_TOKEN env var"}

        # Try loading and running inference
        try:
            t0 = time.time()
            loaded = self.loader.load(entry.name)
            load_time = time.time() - t0

            # Run actual inference
            t1 = time.time()
            infer_result = self._infer(entry, loaded)
            infer_time = time.time() - t1

            if infer_result["ok"]:
                detail = f"load={load_time:.1f}s infer={infer_time:.1f}s {infer_result.get('output', '')}"
                return {"status": "OK", "detail": detail}
            else:
                return {"status": "FAIL", "detail": infer_result.get("error", "inference failed")}

        except Exception as e:
            return {"status": "FAIL", "detail": str(e)[:80]}
        finally:
            self.loader.unload(entry.name)

    def _infer(self, entry: ModelEntry, loaded: dict) -> dict:
        """Run real inference on a model."""
        framework = loaded.get("framework", "")
        text = "Find information about John Smith associated with Acme in Delhi during 2023."

        try:
            if entry.name in ("bert-base-uncased", "deberta-v3-base", "roberta-base", "xlm-roberta-base"):
                return self._infer_transformers_nlp(loaded, text)
            elif entry.name == "gliner-base-v2.1":
                return self._infer_gliner(loaded, text)
            elif entry.name in ("all-MiniLM-L6-v2", "all-mpnet-base-v2"):
                return self._infer_sentence_transformer(loaded)
            elif "clip" in entry.name:
                return self._infer_clip(loaded)
            elif entry.name == "yolo-v8n":
                return self._infer_yolo(loaded)
            elif entry.name == "insightface":
                return self._infer_insightface(loaded)
            elif entry.name == "whisper-base":
                return self._infer_whisper(loaded)
            elif entry.name == "wav2vec2-base":
                return self._infer_wav2vec2(loaded)
            elif entry.name == "pyannote-segmentation":
                return {"ok": True, "output": "loaded (requires audio for inference)"}
            elif "layoutlm" in entry.name:
                return self._infer_layoutlm(loaded)
            else:
                return {"ok": True, "output": "loaded"}
        except Exception as e:
            return {"ok": False, "error": str(e)[:80]}

    def _infer_transformers_nlp(self, loaded: dict, text: str) -> dict:
        import torch
        model = loaded["model"]
        tokenizer = loaded["tokenizer"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
        return {"ok": True, "output": f"shape={list(emb.shape)}"}

    def _infer_gliner(self, loaded: dict, text: str = "") -> dict:
        model = loaded["model"]
        text = text or "Find information about John Smith associated with Acme in Delhi during 2023."
        labels = ["PERSON", "ORGANIZATION", "LOCATION", "DATE"]
        entities = model.predict_entities(text, labels, threshold=0.3)
        names = [e["text"] for e in entities[:5]]
        return {"ok": True, "output": f"entities={names}"}

    def _infer_sentence_transformer(self, loaded: dict) -> dict:
        model = loaded["model"]
        sentences = [
            "Find information about John Smith",
            "Who is John Smith associated with Acme?",
        ]
        embeddings = model.encode(sentences)
        import numpy as np
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return {"ok": True, "output": f"dim={embeddings.shape[1]} sim={similarity:.3f}"}

    def _infer_clip(self, loaded: dict) -> dict:
        import torch
        model = loaded["model"]
        tokenizer = loaded["tokenizer"]
        text = ["a photo of a person", "a photo of a building"]
        tokens = tokenizer(text)
        with torch.no_grad():
            text_features = model.encode_text(tokens)
        return {"ok": True, "output": f"dim={list(text_features.shape)}"}

    def _infer_yolo(self, loaded: dict) -> dict:
        model = loaded["model"]
        import numpy as np
        # Create a blank test image
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(img, verbose=False)
        return {"ok": True, "output": f"detections={len(results[0].boxes)} on blank image"}

    def _infer_insightface(self, loaded: dict) -> dict:
        import numpy as np
        app = loaded["model"]
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        faces = app.get(img)
        return {"ok": True, "output": f"faces={len(faces)} on blank image"}

    def _infer_whisper(self, loaded: dict) -> dict:
        model = loaded["model"]
        # Create a silent audio segment
        import numpy as np
        audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        result = model.transcribe(audio, fp16=False)
        return {"ok": True, "output": f"segments={len(result['segments'])}"}

    def _infer_wav2vec2(self, loaded: dict) -> dict:
        import torch
        import numpy as np
        model = loaded["model"]
        processor = loaded["processor"]
        audio = np.zeros(16000, dtype=np.float32)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        return {"ok": True, "output": f"hidden={list(outputs.last_hidden_state.shape)}"}

    def _infer_layoutlm(self, loaded: dict, text: str = "") -> dict:
        import torch
        model = loaded["model"]
        text = text or "Invoice from Acme Corp dated 2023-01-15 for $5000"
        words = text.split()
        n_words = len(words)
        # Generate fake word-level bounding boxes
        word_boxes = [[int(i*1000/n_words), 10, int((i+1)*1000/n_words), 90] for i in range(n_words)]
        # For LayoutLMv3, use the processor if available; for LayoutLM, use tokenizer+bbox
        if "processor" in loaded:
            processor = loaded["processor"]
            # LayoutLMv3 requires both image + words; provide a blank image
            from PIL import Image as PILImage
            blank_img = PILImage.new('RGB', (224, 224), 'white')
            inputs = processor(
                images=blank_img, text=text,
                return_tensors="pt", truncation=True, max_length=128,
            )
        elif "tokenizer" in loaded:
            tokenizer = loaded["tokenizer"]
            # LayoutLM tokenizer needs explicit word boxes
            enc = tokenizer(
                words, is_split_into_words=True,
                return_tensors="pt", truncation=True, max_length=128,
            )
            seq_len = enc["input_ids"].shape[1]
            boxes = [[0,0,0,0]] + word_boxes + [[0,0,0,0]]
            while len(boxes) < seq_len:
                boxes.append([0,0,0,0])
            boxes = boxes[:seq_len]
            enc["bbox"] = torch.tensor([boxes])
            inputs = enc
        with torch.no_grad():
            outputs = model(**inputs)
        return {"ok": True, "output": f"output_shape={list(outputs.last_hidden_state.shape)}"}

    def _check_opencv(self) -> dict:
        try:
            import cv2
            import numpy as np
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return {"status": "OK", "detail": f"cv2={cv2.__version__} gray={gray.shape}"}
        except Exception as e:
            return {"status": "FAIL", "detail": str(e)[:80]}

    def _check_tesseract(self) -> dict:
        import shutil, os
        # Auto-detect Tesseract on Windows
        if not shutil.which("tesseract"):
            tess = r"C:\Program Files\Tesseract-OCR"
            if os.path.isdir(tess):
                os.environ["PATH"] = tess + ";" + os.environ.get("PATH", "")
        if shutil.which("tesseract"):
            try:
                import pytesseract
                version = pytesseract.get_tesseract_version()
                return {"status": "OK", "detail": f"v{version}", "available": True}
            except Exception as e:
                return {"status": "FAIL", "detail": str(e)[:60], "available": False}
        else:
            return {"status": "UNAVAILABLE", "detail": "Tesseract binary not installed", "available": False}


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Sweep Intelligence Health Check")
    parser.add_argument("--model", help="Check a specific model")
    parser.add_argument("--category", help="Check models in a category")
    args = parser.parse_args()

    hc = HealthCheck()

    if args.model:
        entry = hc.registry.get(args.model)
        if not entry:
            print(f"Unknown model: {args.model}")
            print(f"Available: {', '.join(e.name for e in hc.registry.all())}")
            sys.exit(1)
        result = hc._check_model(entry)
        print(f"\n{entry.name}: {result['status']} — {result.get('detail', '')}")
    elif args.category:
        entries = hc.registry.by_category(args.category)
        if not entries:
            print(f"No models in category: {args.category}")
            sys.exit(1)
        results = {}
        for entry in entries:
            results[entry.name] = hc._check_model(entry)
        for name, r in results.items():
            print(f"  [{r['status']:>10}] {name} — {r.get('detail', '')}")
    else:
        hc.run_all()


if __name__ == "__main__":
    main()
