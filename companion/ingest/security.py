"""Security guards for the ingestion engine.

Ingested content is **untrusted** by definition. Three defenses live here:

* **SSRF guard** — every outbound URL is validated before a connector fetches
  it: only ``http(s)`` schemes, literal private/loopback/link-local/cloud
  metadata IPs are rejected, and hostnames are resolved to check every
  resulting address. Blocks DNS-rebinding style retargeting.
* **Prompt-injection defense** — ingested text is treated as data. Content that
  reads like an instruction override is flagged so the pipeline can reject it,
  and everything that reaches the LLM is wrapped as untrusted quoted material
  (the brain's ``wrap_untrusted`` convention).
* **Sanitization** — control characters stripped, whitespace collapsed, sizes
  capped, and any embedded credentials stripped from stored URLs.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from typing import Optional
from urllib.parse import urlparse

# ─────────────────────────────────────────────────────────────────────────
#  Sanitization
# ─────────────────────────────────────────────────────────────────────────

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"[ \t]+")
_NEWLINE_RE = re.compile(r"\n{3,}")
_BAD_URL_RE = re.compile(r"(?i)[\s]+|['\"\\]")


def sanitize_content(text: str, *, max_chars: int = 60_000) -> str:
    """Collapse control/whitespace noise and cap the length of a document."""
    if not text:
        return ""
    cleaned = _CONTROL_RE.sub("", text)
    cleaned = _NEWLINE_RE.sub("\n\n", cleaned)
    cleaned = _WS_RE.sub(" ", cleaned)
    cleaned = cleaned.strip()
    if max_chars and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned


def strip_credentials(url: str) -> str:
    """Remove user:password from a URL before it is stored/logged."""
    try:
        parsed = urlparse(url)
        if parsed.username is None and parsed.password is None:
            return url
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
        netloc = f"{host}{port}"
        return parsed._replace(netloc=netloc).geturl()
    except Exception:  # noqa: BLE001 - malformed URLs are returned as-is
        return url


# ─────────────────────────────────────────────────────────────────────────
#  SSRF guard
# ─────────────────────────────────────────────────────────────────────────

# Link-local + cloud metadata endpoints are the classic SSRF targets.
_BLOCKED_PREFIXES = (
    "169.254.",   # link-local incl. AWS/GCP/Azure metadata
    "10.",
    "127.",
    "0.",
)


class SSRFBlockedError(ValueError):
    """Raised when a URL fails the outbound-connection guard."""


def _is_blocked_ip(ip_str: str) -> bool:
    ip_str = ip_str.split("%")[0].strip()
    if ip_str.startswith(_BLOCKED_PREFIXES):
        return True
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # unparseable → treat as unsafe
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_outbound_url(url: str, *, resolve: bool = True) -> str:
    """Validate a URL for outbound fetching; raise :class:`SSRFBlockedError`.

    * Rejects non-http(s) schemes.
    * Rejects URLs that embed credentials (we never send them anywhere).
    * Rejects literal private/loopback/link-local/reserved IPs.
    * When ``resolve`` is on, resolves hostnames and rejects the URL if any
      resulting address is blocked (defense against DNS rebinding).
    """
    if not url or "://" not in url:
        raise SSRFBlockedError(f"URL must include a scheme: {url!r}")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise SSRFBlockedError(f"Only http/https schemes are allowed: {url!r}")
    host = parsed.hostname
    if not host:
        raise SSRFBlockedError(f"URL has no host: {url!r}")
    if _BAD_URL_RE.search(url):
        raise SSRFBlockedError(f"URL contains invalid characters")
    if parsed.username is not None or parsed.password is not None:
        raise SSRFBlockedError("URLs with embedded credentials are rejected")

    is_literal_ip = _looks_like_ip(host)
    if is_literal_ip:
        if _is_blocked_ip(host):
            raise SSRFBlockedError(f"Blocked private/link-local address: {host!r}")
        return url

    if resolve:
        try:
            addresses = {
                item[4][0] for item in socket.getaddrinfo(host, None, socket.AF_INET)
            }
        except socket.gaierror as exc:
            raise SSRFBlockedError(f"Could not resolve host {host!r}") from exc
        for addr in addresses:
            if _is_blocked_ip(addr):
                raise SSRFBlockedError(
                    f"Host {host!r} resolves to blocked address {addr!r}"
                )
    return url


def _looks_like_ip(host: str) -> bool:
    if ":" in host and host.count(".") != 3:
        # IPv6 literal
        try:
            ipaddress.ip_address(host)
            return True
        except ValueError:
            return False
    parts = host.split(".")
    if len(parts) != 4:
        return False
    return all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)


# ─────────────────────────────────────────────────────────────────────────
#  Prompt-injection defense
# ─────────────────────────────────────────────────────────────────────────

# Patterns that indicate external text is trying to act as instructions.
# These are intentionally conservative — a hit flags the content for review /
# rejection; it is not a guarantee of malice on its own.
_INJECTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\bignore (all |any |the )?(previous|above|prior) (instructions|prompt|rules)\b"), "instruction override"),
    (re.compile(r"(?i)\bdisregard (all |any |the )?(previous|prior|above|system) (instructions|prompt|rules|messages)\b"), "instruction override"),
    (re.compile(r"(?i)\byou are now (an |a )?(unrestricted|jailbroken|free|no constraints|not bound by)\b"), "jailbreak"),
    (re.compile(r"(?i)\byou (must|have to|need to) (ignore|forget|override) your (system |core )?(instructions|rules|prompt)\b"), "instruction override"),
    (re.compile(r"(?i)<(system|developer|user|assistant)( message| instruction| prompt)?>"), "role tag"),
    (re.compile(r"(?i)\b(system|developer) (prompt|instructions?|message)s?: ?\n"), "role label"),
    (re.compile(r"(?i)\brecursively\b"), "recursion hint"),
    (re.compile(r"(?i)\bunlock the gate\b"), "jailbreak"),
    (re.compile(r"(?i)\boutput (everything |all )?above (in |as )?(base64|markdown|json)\b"), "exfiltration demand"),
)

# Contextual trigger words that make an otherwise-normal document suspect.
_SUSPICIOUS_TERMS = (
    "instructions",
    "jailbreak",
    "reveal system",
    "release your prompt",
    "security policy",
    "bypass",
)


def injection_signals(text: str) -> list[str]:
    """Return the matched injection-pattern labels (empty = looks clean)."""
    if not text:
        return []
    signals: list[str] = []
    for pattern, label in _INJECTION_PATTERNS:
        if pattern.search(text):
            signals.append(label)
    if not signals:
        lowered = text.lower()
        for term in _SUSPICIOUS_TERMS:
            if term in lowered:
                signals.append(f"term:{term}")
                break
    return signals


def looks_like_instruction(text: str) -> bool:
    """True when the content should not be trusted as pure data."""
    return bool(injection_signals(text))
