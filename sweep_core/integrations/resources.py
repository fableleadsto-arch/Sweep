"""Curated external resources and non-installable tool references.

Some items in Sweep's integration list are datasets, platform-specific
projects, or binaries for other toolchains. They are tracked here so
the roadmap stays visible inside the service itself.
"""

from __future__ import annotations

from typing import Any

VOICE_DATASETS: list[dict[str, str]] = [
    {
        "name": "voice_datasets",
        "url": "https://github.com/jim-schwoebel/voice_datasets",
        "note": "curated index of 95+ voice/sound datasets; source for training corpora",
    },
]

DEFERRED_TOOLS: list[dict[str, str]] = [
    {
        "name": "obscura",
        "url": "https://github.com/h4ckf0r0day/obscura",
        "reason": "Rust engine; requires a Rust toolchain not present on this host",
    },
    {
        "name": "google-maps-scraper",
        "url": "https://github.com/gosom/google-maps-scraper",
        "reason": "Go binary; install a release executable when Google Maps ingestion is scheduled",
    },
    {
        "name": "deepspeech",
        "url": "https://github.com/mozilla/DeepSpeech",
        "reason": "archived upstream; superseded by Vosk + Whisper integrations",
    },
    {
        "name": "ultravox",
        "url": "https://github.com/fixie-ai/ultravox",
        "reason": "GPU-scale realtime voice LLM; consume via hosted API instead of local install",
    },
    {
        "name": "bluez",
        "url": "https://github.com/bluez/bluez",
        "reason": "Linux kernel stack; used implicitly through Bleak on Linux hosts",
    },
    {
        "name": "nerf_pl",
        "url": "https://github.com/kwea123/nerf_pl",
        "reason": "Neural Radiance Fields; requires CUDA GPU + conda + PyTorch (Linux/CUDA only)",
    },
    {
        "name": "instant-ngp",
        "url": "https://github.com/NVlabs/instant-ngp",
        "reason": "NVIDIA instant neural graphics primitives; requires CUDA + cmake + C++ build (Linux/CUDA only)",
    },
    {
        "name": "nvidia_canary_1b",
        "url": "https://huggingface.co/nvidia/canary-1b",
        "reason": "1B-parameter multilingual STT; install via huggingface_hub.snapshot_download when GPU is available (~3GB)",
    },
    {
        "name": "insightface",
        "url": "https://github.com/deepinsight/insightface",
        "reason": "requires native build compiler; DeepFace ArcFace backend covers same functionality via TF/ONNX",
    },
    {
        "name": "face_recognition",
        "url": "https://github.com/ageitgey/face_recognition",
        "reason": "requires cmake + dlib compilation; OpenCV YuNet + DeepFace cover face detection/recognition",
    },
]


def availability() -> dict[str, Any]:
    return {
        "voice_datasets_indexed": len(VOICE_DATASETS),
        "deferred_tools_tracked": len(DEFERRED_TOOLS),
    }
