# Sweep Intelligence — Installation Report

## Summary

```
Total models registered:    18
Models verified (OK):       14
Models unavailable:          4
Models failed:               0

CPU detected:               Yes (no GPU)
RAM detected:               16.9GB
Storage used:               17.4GB

Local-only loading:         PASS (all verified models load from models/)
Intent Core integration:    PASS
Health check:               PASS (14/14 downloaded models)
```

## Models Verified — Real Inference

### NLP (5/5)
| Model | Checkpoint | Load | Infer | Output |
|-------|-----------|------|-------|--------|
| BERT | bert-base-uncased | 68.0s | 4.7s | shape=[1, 768] |
| DeBERTa-v3 | microsoft/deberta-v3-base | 9.5s | 2.5s | shape=[1, 768] |
| RoBERTa | FacebookAI/roberta-base | 1.7s | 2.7s | shape=[1, 768] |
| XLM-RoBERTa | FacebookAI/xlm-roberta-base | 4.9s | 2.8s | shape=[1, 768] |
| GLiNER | urchade/gliner_medium-v2.1 | 78.4s | 2.9s | entities=['John Smith', 'Acme', 'Delhi', '2023'] |

### Embeddings (2/2)
| Model | Checkpoint | Load | Infer | Output |
|-------|-----------|------|-------|--------|
| MiniLM-L6 | all-MiniLM-L6-v2 | 2.8s | 1.0s | dim=384, sim=0.726 |
| MPNet | all-mpnet-base-v2 | 1.2s | 4.1s | dim=768, sim=0.645 |

### Vision (2/3)
| Model | Checkpoint | Load | Infer | Output |
|-------|-----------|------|-------|--------|
| CLIP B/32 | openai/clip-vit-base-patch32 | 18.6s | 0.5s | dim=[2, 512] |
| CLIP L/14 | — | — | — | UNAVAILABLE (requires GPU) |
| YOLO-v8n | yolov8n.pt | 3.8s | 3.3s | 0 detections on blank |

### Face (1/1)
| Model | Checkpoint | Load | Infer | Output |
|-------|-----------|------|-------|--------|
| InsightFace | buffalo_l (5 ONNX models) | 10.8s | 1.0s | 0 faces on blank |

### Audio/Speech (2/3)
| Model | Checkpoint | Load | Infer | Output |
|-------|-----------|------|-------|--------|
| Whisper-base | base | 7.7s | 14.7s | 0 segments on silent |
| Wav2Vec2 | facebook/wav2vec2-base | 4.1s | 2.9s | hidden=[1, 49, 768] |
| pyannote | — | — | — | UNAVAILABLE (requires HF_TOKEN) |

### Documents (1/3)
| Model | Checkpoint | Load | Infer | Output |
|-------|-----------|------|-------|--------|
| LayoutLM | microsoft/layoutlm-base-uncased | 1.3s | 3.8s | output_shape=[1, 19, 768] |
| LayoutLMv3 | — | — | — | UNAVAILABLE (requires Tesseract binary) |
| Tesseract | system binary | — | — | UNAVAILABLE (not installed on this machine) |

### Engines (1/1)
| Engine | Status | Output |
|--------|--------|--------|
| OpenCV | cv2=5.0.0 | image load/convert/resize OK |

## Unavailable Models — Remediation

| Model | Reason | Fix |
|-------|--------|-----|
| CLIP L/14 | Requires GPU (optional) | Install CUDA GPU, then run `--include-optional` |
| pyannote | Requires HuggingFace auth | Set `HF_TOKEN` env var, accept license at huggingface.co |
| LayoutLMv3 | Requires Tesseract binary | `winget install UB-Mannheim.TesseractOCR` |
| Tesseract | Not installed | `winget install UB-Mannheim.TesseractOCR` |

## Storage Breakdown

```
models/nlp/          11.77GB  (5 models)
models/embeddings/    2.23GB  (2 models)
models/documents/     1.91GB  (1 model)
models/vision/        0.61GB  (1 model)
models/audio/         0.38GB  (1 model)
models/face/          0.34GB  (1 model)
models/speech/        0.15GB  (1 model)
models/detection/     0.01GB  (1 model)
─────────────────────────────
Total:               17.39GB
```

## File Structure

```
services/intelligence/
├── __init__.py
├── download_models.py      # CLI: --core/--vision/--audio/--all
├── download_all.py         # One-shot batch downloader
├── download_targeted.py    # Targeted downloader (skip TF/Flax)
├── health_check.py         # Real inference verification
├── model_downloader.py     # Download engine
├── intent_core.py          # Intent Core pipeline
├── requirements-core.txt
├── requirements-vision.txt
├── requirements-audio.txt
├── requirements-documents.txt
├── requirements-advanced.txt
├── REPORT.md
├── models/
│   ├── __init__.py
│   ├── registry.json       # Download/verification status
│   └── capabilities.yaml   # Task → model mapping
├── model_manager/
│   ├── __init__.py
│   ├── device.py           # CPU/GPU/RAM detection
│   ├── registry.py         # Model registry (18 entries)
│   ├── loader.py           # Lazy model loading
│   └── cache.py            # LRU model cache
├── vision/
│   ├── __init__.py
│   └── opencv_engine.py    # OpenCV processing
├── document/
│   ├── __init__.py
│   └── ocr_engine.py       # Tesseract wrapper
└── tests/
    └── __init__.py

models/                      # Downloaded weights (17.4GB)
├── nlp/                     # BERT, DeBERTa, RoBERTa, XLM-R, GLiNER
├── embeddings/              # MiniLM-L6, MPNet
├── vision/                  # CLIP B/32
├── detection/               # YOLO-v8n
├── face/                    # InsightFace buffalo_l
├── speech/                  # Whisper base.pt
├── audio/                   # Wav2Vec2
├── documents/               # LayoutLM
├── multimodal/              # (empty, ready for Qwen3-VL)
└── indexes/                 # (empty, ready for FAISS)
```

## Installed Packages

```
torch                   2.9.1+cu126
tensorflow              2.21.0
transformers            4.57.0
sentence-transformers   5.1.2
open-clip-torch         3.3.0
ultralytics             8.4.128
insightface             1.0.1
openai-whisper          20250625
pyannote.audio          4.0.7
gliner                  0.2.28
faiss-cpu               1.15.0
opencv-python           5.0.0.93
onnxruntime             1.23.2
pytesseract             0.3.13
tf-keras                2.21.0
```

## Intent Core Integration

The `IntentCore` class provides the standardized pipeline:

```python
from services.intelligence.intent_core import IntentCore

core = IntentCore()

# Text → entities + embeddings
core.process("Find John Smith at Acme in Delhi")
core.extract_entities("Find John Smith at Acme in Delhi 2023")
core.embed_text("Find John Smith at Acme in Delhi 2023")

# Image → objects + embeddings
core.detect_objects("photo.jpg")
core.embed_image("photo.jpg")

# Audio → transcription
core.transcribe("recording.wav")

# Vector search
emb = core.embed_text("query")
results = core.search_similar(emb, candidate_embeddings, top_k=5)
```

## Commands

```bash
# Check all models
python services/intelligence/health_check.py

# Download specific category
python services/intelligence/download_models.py --core
python services/intelligence/download_models.py --vision
python services/intelligence/download_models.py --audio
python services/intelligence/download_models.py --all

# List model registry
python services/intelligence/download_models.py --list
```
