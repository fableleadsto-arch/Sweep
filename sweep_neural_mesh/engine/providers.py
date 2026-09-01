"""
Provider Abstraction Layer — original Sweep interfaces for all model capabilities.

Each provider is replaceable. Sweep's core logic does not care whether inference
happens on CPU, local GPU, cloud GPU, or another compatible backend.

Third-party models are used as legitimate dependencies through these interfaces.
Sweep's orchestration, routing, and verification logic is independently implemented.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("sweep.providers")


# ════════════════════════════════════════════════════════════════════
# RESULT TYPES
# ════════════════════════════════════════════════════════════════════

class ProviderStatus(Enum):
    READY = "ready"
    LOADING = "loading"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


@dataclass
class ProviderResult:
    """Generic result from any provider."""
    output: Any
    confidence: float = 0.0
    latency_ms: float = 0.0
    provider: str = ""
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    status: ProviderStatus = ProviderStatus.READY
    error: str | None = None


@dataclass
class EmbeddingResult:
    """Result from an embedding provider."""
    vector: list[float]
    dimension: int = 0
    model: str = ""
    latency_ms: float = 0.0


@dataclass
class SimilarityResult:
    """Result from a similarity comparison."""
    score: float
    direction: str = "neutral"  # supports, refutes, neutral
    latency_ms: float = 0.0


# ════════════════════════════════════════════════════════════════════
# PROVIDER INTERFACES (original Sweep abstractions)
# ════════════════════════════════════════════════════════════════════

class LanguageProvider(ABC):
    """Interface for language understanding and generation."""

    @abstractmethod
    def initialize(self) -> bool:
        """Load model. Returns True if successful."""
        ...

    @abstractmethod
    def answer(self, query: str, context: list[str] | None = None) -> ProviderResult:
        """Answer a question given optional context."""
        ...

    @abstractmethod
    def summarize(self, text: str, max_length: int = 200) -> ProviderResult:
        """Summarize text."""
        ...

    @abstractmethod
    def classify(self, text: str, categories: list[str]) -> ProviderResult:
        """Classify text into one of the given categories."""
        ...

    @abstractmethod
    def extract(self, text: str, schema: dict | None = None) -> ProviderResult:
        """Extract structured information from text."""
        ...

    @property
    @abstractmethod
    def status(self) -> ProviderStatus:
        """Current provider status."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the underlying model."""
        ...


class VisionProvider(ABC):
    """Interface for image understanding."""

    @abstractmethod
    def initialize(self) -> bool:
        """Load model. Returns True if successful."""
        ...

    @abstractmethod
    def ocr(self, image_path: str) -> ProviderResult:
        """Extract text from image."""
        ...

    @abstractmethod
    def embed(self, image_path: str) -> EmbeddingResult:
        """Get image embedding vector."""
        ...

    @abstractmethod
    def similarity(self, image_a: str, image_b: str) -> SimilarityResult:
        """Compare two images for similarity."""
        ...

    @abstractmethod
    def describe(self, image_path: str) -> ProviderResult:
        """Describe image contents."""
        ...

    @property
    @abstractmethod
    def status(self) -> ProviderStatus:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...


class AudioProvider(ABC):
    """Interface for audio understanding."""

    @abstractmethod
    def initialize(self) -> bool:
        ...

    @abstractmethod
    def transcribe(self, audio_path: str) -> ProviderResult:
        """Transcribe audio to text."""
        ...

    @abstractmethod
    def embed(self, audio_path: str) -> EmbeddingResult:
        """Get audio embedding."""
        ...

    @property
    @abstractmethod
    def status(self) -> ProviderStatus:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...


class EmbeddingProvider(ABC):
    """Interface for text embeddings."""

    @abstractmethod
    def initialize(self) -> bool:
        ...

    @abstractmethod
    def embed(self, text: str) -> EmbeddingResult:
        """Get text embedding."""
        ...

    @abstractmethod
    def similarity(self, text_a: str, text_b: str) -> SimilarityResult:
        """Semantic similarity between two texts."""
        ...

    @abstractmethod
    def batch_embed(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed multiple texts."""
        ...

    @property
    @abstractmethod
    def status(self) -> ProviderStatus:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension."""
        ...


class RetrievalProvider(ABC):
    """Interface for information retrieval."""

    @abstractmethod
    def initialize(self) -> bool:
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> ProviderResult:
        """Search for relevant documents."""
        ...

    @abstractmethod
    def index(self, documents: list[dict]) -> ProviderResult:
        """Index documents for retrieval."""
        ...

    @property
    @abstractmethod
    def status(self) -> ProviderStatus:
        ...


# ════════════════════════════════════════════════════════════════════
# PROVIDER REGISTRY (original Sweep implementation)
# ════════════════════════════════════════════════════════════════════

class ProviderRegistry:
    """Manages providers with lazy loading and health monitoring.

    Sweep-original implementation. Third-party models are loaded through
    provider interfaces — Sweep's core logic never depends on specific models.
    """

    def __init__(self) -> None:
        self._providers: dict[str, Any] = {}
        self._factories: dict[str, callable] = {}
        self._loaded: dict[str, bool] = {}
        self._last_health: dict[str, float] = {}

    def register(self, name: str, factory: callable) -> None:
        """Register a provider factory (lazy — not loaded yet)."""
        self._factories[name] = factory
        self._loaded[name] = False
        logger.info(f"Registered provider: {name}")

    def get(self, name: str) -> Any:
        """Get a provider, loading it lazily if needed."""
        if name not in self._factories:
            raise KeyError(f"Provider '{name}' not registered")

        if not self._loaded.get(name, False):
            logger.info(f"Loading provider: {name}")
            t0 = time.perf_counter()
            try:
                self._providers[name] = self._factories[name]()
                self._loaded[name] = True
                latency = (time.perf_counter() - t0) * 1000
                self._last_health[name] = time.time()
                logger.info(f"Loaded {name} in {latency:.0f}ms")
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
                raise

        return self._providers[name]

    def unload(self, name: str) -> None:
        """Unload a provider to free memory."""
        if name in self._providers:
            del self._providers[name]
            self._loaded[name] = False
            logger.info(f"Unloaded provider: {name}")

    def unload_all(self) -> None:
        """Unload all providers."""
        for name in list(self._providers.keys()):
            self.unload(name)

    def status(self) -> dict[str, dict]:
        """Get status of all registered providers."""
        result = {}
        for name in self._factories:
            result[name] = {
                "loaded": self._loaded.get(name, False),
                "last_health": self._last_health.get(name),
            }
        return result

    def health_check(self) -> dict[str, bool]:
        """Check health of all loaded providers."""
        results = {}
        for name, provider in self._providers.items():
            try:
                if hasattr(provider, "status"):
                    results[name] = provider.status == ProviderStatus.READY
                else:
                    results[name] = True
            except Exception:
                results[name] = False
        return results


# ════════════════════════════════════════════════════════════════════
# CPU-OPTIMIZED PROVIDERS (original Sweep implementations)
# ════════════════════════════════════════════════════════════════════

class MiniLMEmbeddingProvider(EmbeddingProvider):
    """CPU-optimized embedding provider using all-MiniLM-L6-v2.

    Uses sentence-transformers (Apache 2.0 license) as a dependency.
    Sweep's wrapping, caching, and integration logic is original.
    """

    def __init__(self) -> None:
        self._model = None
        self._model_name = "all-MiniLM-L6-v2"
        self._dim = 384
        self._cache: dict[str, EmbeddingResult] = {}

    def initialize(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            logger.info(f"Loaded {self._model_name} ({self._dim}d)")
            return True
        except ImportError:
            logger.warning("sentence-transformers not available")
            return False
        except Exception as e:
            logger.error(f"Failed to load {self._model_name}: {e}")
            return False

    @property
    def status(self) -> ProviderStatus:
        return ProviderStatus.READY if self._model else ProviderStatus.UNAVAILABLE

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> EmbeddingResult:
        if text in self._cache:
            return self._cache[text]
        if not self._model:
            self.initialize()
        t0 = time.perf_counter()
        vec = self._model.encode(text).tolist()
        result = EmbeddingResult(
            vector=vec, dimension=self._dim,
            model=self._model_name,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        self._cache[text] = result
        return result

    def similarity(self, text_a: str, text_b: str) -> SimilarityResult:
        import numpy as np
        ea = self.embed(text_a)
        eb = self.embed(text_b)
        score = float(np.dot(ea.vector, eb.vector) / (
            np.linalg.norm(ea.vector) * np.linalg.norm(eb.vector) + 1e-8
        ))
        direction = "supports" if score > 0.5 else "refutes" if score < -0.3 else "neutral"
        return SimilarityResult(score=score, direction=direction)

    def batch_embed(self, texts: list[str]) -> list[EmbeddingResult]:
        return [self.embed(t) for t in texts]
