"""Download external assets for Sweep integrations (idempotent).

Fetches:
  * Vosk small English model -> models/vosk/vosk-model-small-en-us-0.15
  * Meilisearch Windows binary -> bin/meilisearch.exe

Run:  python scripts/fetch_integration_assets.py
"""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

VOSK_MODEL_URL = (
    "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
)
MEILI_RELEASE_URL = (
    "https://github.com/meilisearch/meilisearch/releases/latest/download/"
    "meilisearch-windows-amd64.exe"
)


def fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip (exists): {dest}")
        return dest
    print(f"downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def main() -> int:
    root = Path(__file__).resolve().parents[1]

    vosk_zip = fetch(VOSK_MODEL_URL, root / "models" / "vosk-model-small-en-us-0.15.zip")
    target = root / "models" / "vosk"
    if not any(target.glob("vosk-model*")) and not (target / "vosk-model-small-en-us-0.15").exists():
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(vosk_zip) as zf:
            zf.extractall(target)
        print(f"extracted to {target}")
    else:
        print("skip (exists): vosk model directory")

    meili = fetch(MEILI_RELEASE_URL, root / "bin" / "meilisearch.exe")
    meili.chmod(0o755)
    print(f"ready: {meili}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
