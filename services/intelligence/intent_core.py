"""Intent Core — multimodal perception pipeline for Sweep.

Routes input through:
  INPUT → MODALITY DETECTION → MODEL ROUTING → INFERENCE → STANDARDIZED OBSERVATIONS
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .model_manager.device import DeviceInfo, detect_device
from .model_manager.loader import ModelLoader
from .model_manager.registry import ModelRegistry

log = logging.getLogger("sweep.intent_core")


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"
    VIDEO = "video"
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"


@dataclass
class Observation:
    """Standardized output from any model inference."""
    modality: str
    model: str
    task: str
    data: dict
    confidence: float = 0.0
    entities: list[dict] = field(default_factory=list)
    embeddings: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    latency_ms: float = 0

    def to_dict(self) -> dict:
        return {
            "modality": self.modality,
            "model": self.model,
            "task": self.task,
            "data": self.data,
            "confidence": self.confidence,
            "entities": self.entities,
            "embedding_dim": len(self.embeddings),
            "metadata": self.metadata,
            "latency_ms": self.latency_ms,
        }


@dataclass
class InvestigationContext:
    """Accumulated intelligence from processing."""
    observations: list[Observation] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    intent: str = ""
    objective: str = ""

    def add_observation(self, obs: Observation):
        self.observations.append(obs)
        self.entities.extend(obs.entities)


class IntentCore:
    """Routes multimodal input through the model stack."""

    def __init__(self, device: DeviceInfo | None = None):
        self.device = device or detect_device()
        self.registry = ModelRegistry()
        self.loader = ModelLoader(self.registry, self.device)
        self._context = InvestigationContext()

    @property
    def context(self) -> InvestigationContext:
        return self._context

    def detect_modality(self, input_data: Any) -> Modality:
        """Detect the modality of input data."""
        if isinstance(input_data, str):
            # Could be text or file path
            import os
            if os.path.isfile(input_data):
                ext = os.path.splitext(input_data)[1].lower()
                if ext in (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"):
                    return Modality.IMAGE
                if ext in (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"):
                    return Modality.AUDIO
                if ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
                    return Modality.VIDEO
                if ext in (".pdf", ".docx", ".doc", ".xlsx", ".pptx"):
                    return Modality.DOCUMENT
                if ext in (".txt", ".md", ".html", ".json", ".csv"):
                    return Modality.TEXT
                return Modality.UNKNOWN
            return Modality.TEXT
        if isinstance(input_data, bytes):
            return Modality.IMAGE  # heuristic
        return Modality.UNKNOWN

    def process(self, input_data: Any) -> InvestigationContext:
        """Process input through the full pipeline."""
        modality = self.detect_modality(input_data)
        log.info(f"Detected modality: {modality.value}")

        if modality == Modality.TEXT:
            self._process_text(input_data)
        elif modality == Modality.IMAGE:
            self._process_image(input_data)
        elif modality == Modality.AUDIO:
            self._process_audio(input_data)
        elif modality == Modality.DOCUMENT:
            self._process_document(input_data)

        return self._context

    def extract_entities(self, text: str) -> list[dict]:
        """Extract entities using GLiNER."""
        t0 = time.time()
        try:
            model_bundle = self.loader.load("gliner-base-v2.1")
            model = model_bundle["model"]
            labels = [
                "PERSON", "ORGANIZATION", "ADDRESS", "CITY", "COUNTRY",
                "DATE", "USERNAME", "DOMAIN", "EMAIL", "PHONE",
                "EVENT", "LOCATION",
            ]
            entities = model.predict_entities(text, labels, threshold=0.3)
            obs = Observation(
                modality="text", model="gliner-base-v2.1",
                task="entity_extraction",
                data={"text": text[:200], "entities": entities},
                entities=[
                    {"text": e["text"], "label": e["label"], "score": e["score"]}
                    for e in entities
                ],
                confidence=sum(e["score"] for e in entities) / max(len(entities), 1),
                latency_ms=(time.time() - t0) * 1000,
            )
            self._context.add_observation(obs)
            return obs.entities
        except Exception as e:
            log.error(f"Entity extraction failed: {e}")
            return []

    def embed_text(self, text: str, model_name: str = "all-MiniLM-L6-v2") -> list[float]:
        """Generate text embeddings."""
        t0 = time.time()
        try:
            model_bundle = self.loader.load(model_name)
            model = model_bundle["model"]
            embedding = model.encode(text, normalize_embeddings=True)
            obs = Observation(
                modality="text", model=model_name,
                task="text_embedding",
                data={"text": text[:200]},
                embeddings=embedding.tolist(),
                confidence=1.0,
                latency_ms=(time.time() - t0) * 1000,
            )
            self._context.add_observation(obs)
            return embedding.tolist()
        except Exception as e:
            log.error(f"Embedding failed: {e}")
            return []

    def embed_image(self, image_path: str) -> list[float]:
        """Generate image embeddings via CLIP."""
        t0 = time.time()
        try:
            model_bundle = self.loader.load("clip-vit-b32")
            import torch
            from PIL import Image
            model = model_bundle["model"]
            preprocess = model_bundle["preprocess"]
            image = Image.open(image_path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0)
            device = self.device.device
            if device == "cuda":
                image_tensor = image_tensor.cuda()
            with torch.no_grad():
                features = model.encode_image(image_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
            obs = Observation(
                modality="image", model="clip-vit-b32",
                task="image_embedding",
                data={"image": image_path},
                embeddings=features[0].cpu().tolist(),
                confidence=1.0,
                latency_ms=(time.time() - t0) * 1000,
            )
            self._context.add_observation(obs)
            return features[0].cpu().tolist()
        except Exception as e:
            log.error(f"Image embedding failed: {e}")
            return []

    def detect_objects(self, image_path: str) -> list[dict]:
        """Detect objects in image via YOLO."""
        t0 = time.time()
        try:
            model_bundle = self.loader.load("yolo-v8n")
            model = model_bundle["model"]
            results = model(image_path, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    detections.append({
                        "class": r.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox": box.xyxy[0].tolist(),
                    })
            obs = Observation(
                modality="image", model="yolo-v8n",
                task="object_detection",
                data={"image": image_path, "detections": detections},
                confidence=max((d["confidence"] for d in detections), default=0),
                latency_ms=(time.time() - t0) * 1000,
            )
            self._context.add_observation(obs)
            return detections
        except Exception as e:
            log.error(f"Object detection failed: {e}")
            return []

    def transcribe(self, audio_path: str) -> dict:
        """Transcribe audio via Whisper."""
        t0 = time.time()
        try:
            model_bundle = self.loader.load("whisper-base")
            model = model_bundle["model"]
            result = model.transcribe(audio_path, fp16=False)
            obs = Observation(
                modality="audio", model="whisper-base",
                task="speech_to_text",
                data={
                    "text": result["text"],
                    "language": result.get("language", ""),
                    "segments": len(result["segments"]),
                },
                confidence=1.0,
                metadata={"language": result.get("language", "")},
                latency_ms=(time.time() - t0) * 1000,
            )
            self._context.add_observation(obs)
            return {"text": result["text"], "language": result.get("language", "")}
        except Exception as e:
            log.error(f"Transcription failed: {e}")
            return {"text": "", "language": ""}

    def search_similar(self, query_embedding: list[float], candidate_embeddings: list[list[float]],
                       top_k: int = 5) -> list[dict]:
        """Find similar items using FAISS."""
        import numpy as np
        try:
            import faiss
            dim = len(query_embedding)
            index = faiss.IndexFlatIP(dim)  # inner product (cosine for normalized vectors)
            embeddings = np.array(candidate_embeddings, dtype=np.float32)
            index.add(embeddings)
            query = np.array([query_embedding], dtype=np.float32)
            scores, indices = index.search(query, min(top_k, len(candidate_embeddings)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    results.append({"index": int(idx), "score": float(score)})
            return results
        except Exception as e:
            log.error(f"FAISS search failed: {e}")
            return []

    def _process_text(self, text: str):
        """Full text processing pipeline."""
        self.extract_entities(text)
        self.embed_text(text)

    def _process_image(self, image_path: str):
        """Full image processing pipeline."""
        self.embed_image(image_path)
        self.detect_objects(image_path)

    def _process_audio(self, audio_path: str):
        """Full audio processing pipeline."""
        self.transcribe(audio_path)

    def _process_document(self, doc_path: str):
        """Full document processing pipeline."""
        try:
            self.extract_entities(open(doc_path).read()[:5000])
        except Exception:
            pass

    def reset(self):
        """Clear accumulated context."""
        self._context = InvestigationContext()
