# Sweep CPU Mode

## Overview

Sweep is designed as a **CPU-first** neural engine. It detects available hardware automatically and selects the appropriate runtime profile.

## Hardware Detection

```
CPU: Detected via os.cpu_count() and platform info
RAM: Detected via psutil or /proc/meminfo
GPU: Detected via torch.cuda.is_available()
VRAM: Detected via torch.cuda.get_device_properties()
```

## Runtime Profiles

### LOW_RESOURCE
- CPU only, limited RAM
- Lightweight models (MiniLM, nano Relay)
- Low concurrency
- Reduced batch sizes

### BALANCED
- Strong CPU, larger RAM
- Optional GPU acceleration
- Full model suite

### HIGH_PERFORMANCE
- Dedicated GPU
- High RAM
- GPU-accelerated inference

## Model Loading Strategy

1. **Lazy loading** — Models loaded on first use, not at startup
2. **Caching** — Models stay loaded until memory pressure
3. **Priority** — Most-used models loaded first
4. **Eviction** — Least-recently-used models unloaded under memory pressure

## Memory Budget

| Component | LOW_RESOURCE | BALANCED | HIGH_PERFORMANCE |
|-----------|-------------|----------|------------------|
| MiniLM | 80MB | 80MB | 80MB |
| Relay small | 42MB | 42MB | 42MB |
| Working memory | 10MB | 50MB | 200MB |
| Evidence memory | 50MB | 200MB | 1GB |
| **Total** | ~182MB | ~372MB | ~1.3GB |

## Graceful Degradation

If a model cannot load:
1. Log the failure
2. Continue with available models
3. Use rule-based fallback
4. Report degraded capability

Sweep must not crash because an optional model cannot load.
