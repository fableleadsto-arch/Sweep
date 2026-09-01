"""Vendored-framework inventory capability.

Relay ships the source of the largest AI/ML frameworks in-repo
(companion/vendor/). This capability is always available — it needs no
external framework — and reports exactly what source is stored locally:

  * importable packages (pure-Python source the engine can import directly)
  * source archives (compiled/CUDA giants stored for offline builds)
  * bundled wheels (pre-built platform wheels like the PyTorch CUDA bundle,
    stored locally; not committed to git because they exceed GitHub's size
    limits)

The agent can use this to answer "which ML frameworks does Relay have source
for?" and to decide whether to import a framework from vendor, build it from
the stored archive, or install the stored pre-built wheel.
"""

from __future__ import annotations

from typing import Any

from .vendor_loader import describe_all


def run_vendor_source(payload: dict[str, Any]) -> dict[str, Any]:
    """Inventory the locally vendored framework source (never raises)."""
    inventory = describe_all()
    importable = inventory["importable"]
    archives = [a for a in inventory["archives"] if a["present"]]
    wheels = [w for w in inventory["wheels"] if w["present"]]
    return {
        "result": {
            "root": inventory["root"],
            "importable_packages": [p["name"] for p in importable],
            "source_archives": [
                {
                    "name": a["name"],
                    "version": a["version"],
                    "license": a["license"],
                    "filename": a["filename"],
                    "path": a["path"],
                }
                for a in archives
            ],
            "bundled_wheels": [
                {
                    "name": w["name"],
                    "version": w["version"],
                    "license": w["license"],
                    "filename": w["filename"],
                    "path": w["path"],
                }
                for w in wheels
            ],
            "counts": {
                "importable": len(importable),
                "archives": len(archives),
                "wheels": len(wheels),
            },
        },
        "summary": (
            f"Relay has {len(importable)} importable vendored framework(s) "
            f"({', '.join(p['name'] for p in importable) or 'none'}), "
            f"{len(archives)} stored source archive(s) "
            f"({', '.join(a['name'] for a in archives) or 'none'}), and "
            f"{len(wheels)} bundled wheel(s) "
            f"({', '.join(w['name'] for w in wheels) or 'none'})."
        ),
        "libraries_used": [],
    }
