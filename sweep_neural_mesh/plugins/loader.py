"""
Plugin System — manifest-based plugin loading and auto-discovery.

Plugins are Python modules or packages that register themselves via
a manifest (dict or YAML-like structure). The loader discovers plugins
in configured directories, validates their manifests, and makes them
available to the Neural Mesh.

Each plugin manifest specifies:
  - name: unique identifier
  - version: semver string
  - capabilities: list of capability strings
  - entry_point: dotted module path
  - priority: routing priority (lower = preferred)
  - metadata: arbitrary key-value pairs

This is an original implementation of a plugin architecture designed
specifically for the Sweep Neural Mesh's capability-based routing.
"""
from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PluginManifest:
    """Describes a plugin's identity and capabilities."""
    name: str
    version: str
    capabilities: list[str] = field(default_factory=list)
    entry_point: str = ""
    priority: int = 100  # lower = preferred
    metadata: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "capabilities": self.capabilities,
            "entry_point": self.entry_point,
            "priority": self.priority,
            "metadata": self.metadata,
            "enabled": self.enabled,
        }


@dataclass
class LoadedPlugin:
    """A plugin that has been imported and is ready to use."""
    manifest: PluginManifest
    module: Any  # the imported module object
    loaded_at: float = 0.0
    error: str | None = None

    @property
    def healthy(self) -> bool:
        return self.error is None and self.module is not None


class PluginLoader:
    """
    Discovers, validates, and loads plugins from manifests.

    Usage:
        loader = PluginLoader()
        loader.register_manifest(PluginManifest(name="fastpath", version="1.0.0", ...))
        loader.discover_directory(Path("sweep_neural_mesh/plugins"))
        plugins = loader.load_all()
    """

    def __init__(self) -> None:
        self._manifests: dict[str, PluginManifest] = {}
        self._loaded: dict[str, LoadedPlugin] = {}
        self._search_dirs: list[Path] = []

    def register_manifest(self, manifest: PluginManifest) -> None:
        """Register a plugin manifest (does not load it yet)."""
        if manifest.name in self._manifests:
            existing = self._manifests[manifest.name]
            if manifest.version != existing.version:
                logger.warning(
                    "plugin %s version conflict: %s -> %s",
                    manifest.name, existing.version, manifest.version,
                )
        self._manifests[manifest.name] = manifest

    def discover_directory(self, directory: Path) -> int:
        """
        Scan a directory for plugin manifests.

        Looks for:
          - sweep_plugin.json files
          - __init__.py files with a SWEEP_PLUGINManifest dict

        Returns the number of new manifests found.
        """
        if not directory.is_dir():
            return 0

        self._search_dirs.append(directory)
        found = 0

        for child in directory.iterdir():
            if child.is_file() and child.name == "sweep_plugin.json":
                try:
                    import json
                    data = json.loads(child.read_text(encoding="utf-8"))
                    manifest = PluginManifest(
                        name=data["name"],
                        version=data.get("version", "0.0.0"),
                        capabilities=data.get("capabilities", []),
                        entry_point=data.get("entry_point", ""),
                        priority=data.get("priority", 100),
                        metadata=data.get("metadata", {}),
                        enabled=data.get("enabled", True),
                    )
                    self.register_manifest(manifest)
                    found += 1
                except Exception as exc:
                    logger.warning("failed to load manifest %s: %s", child, exc)

            elif child.is_dir():
                init_file = child / "__init__.py"
                if init_file.exists():
                    try:
                        source = init_file.read_text(encoding="utf-8")
                        if "SWEEP_PLUGIN" in source:
                            # Extract the dict literal — simple eval-free approach
                            # Plugin authors put: SWEEP_PLUGIN = {"name": ..., ...}
                            manifest_dict = self._extract_manifest_dict(source)
                            if manifest_dict:
                                manifest = PluginManifest(
                                    name=manifest_dict["name"],
                                    version=manifest_dict.get("version", "0.0.0"),
                                    capabilities=manifest_dict.get("capabilities", []),
                                    entry_point=manifest_dict.get("entry_point", child.name),
                                    priority=manifest_dict.get("priority", 100),
                                    metadata=manifest_dict.get("metadata", {}),
                                    enabled=manifest_dict.get("enabled", True),
                                )
                                self.register_manifest(manifest)
                                found += 1
                    except Exception as exc:
                        logger.warning("failed to scan %s: %s", child, exc)

        return found

    def _extract_manifest_dict(self, source: str) -> dict[str, Any] | None:
        """Extract SWEEP_PLUGIN = {...} dict from source without eval."""
        import ast
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("SWEEP_PLUGIN") and "=" in stripped:
                # Find the dict literal after the = sign
                eq_idx = stripped.index("=")
                dict_str = stripped[eq_idx + 1:].strip()
                # Handle trailing comments
                if "#" in dict_str:
                    dict_str = dict_str[:dict_str.index("#")].strip()
                try:
                    return ast.literal_eval(dict_str)
                except (ValueError, SyntaxError):
                    pass
        return None

    def load_plugin(self, name: str) -> LoadedPlugin:
        """Load a single registered plugin by name."""
        if name not in self._manifests:
            raise KeyError(f"unknown plugin: {name}")
        if name in self._loaded and self._loaded[name].healthy:
            return self._loaded[name]

        manifest = self._manifests[name]
        if not manifest.enabled:
            lp = LoadedPlugin(manifest=manifest, module=None, error="disabled")
            self._loaded[name] = lp
            return lp

        t0 = time.perf_counter()
        try:
            module = importlib.import_module(manifest.entry_point)
            lp = LoadedPlugin(
                manifest=manifest,
                module=module,
                loaded_at=(time.perf_counter() - t0) * 1000,
            )
            logger.info("loaded plugin %s v%s in %.1fms", name, manifest.version, lp.loaded_at)
        except Exception as exc:
            lp = LoadedPlugin(
                manifest=manifest,
                module=None,
                loaded_at=(time.perf_counter() - t0) * 1000,
                error=str(exc),
            )
            logger.warning("failed to load plugin %s: %s", name, exc)

        self._loaded[name] = lp
        return lp

    def load_all(self) -> dict[str, LoadedPlugin]:
        """Load all enabled registered plugins. Returns map of name -> LoadedPlugin."""
        results: dict[str, LoadedPlugin] = {}
        for name, manifest in sorted(self._manifests.items(), key=lambda x: x[1].priority):
            if manifest.enabled:
                results[name] = self.load_plugin(name)
        return results

    def plugins_for_capability(self, capability: str) -> list[LoadedPlugin]:
        """Find all loaded plugins that provide a given capability."""
        return [
            lp for lp in self._loaded.values()
            if lp.healthy and capability in lp.manifest.capabilities
        ]

    @property
    def manifest_count(self) -> int:
        return len(self._manifests)

    @property
    def loaded_count(self) -> int:
        return sum(1 for lp in self._loaded.values() if lp.healthy)

    def summary(self) -> dict[str, Any]:
        return {
            "manifests_registered": self.manifest_count,
            "plugins_loaded": self.loaded_count,
            "plugins_failed": sum(
                1 for lp in self._loaded.values() if lp.error is not None
            ),
            "search_dirs": [str(d) for d in self._search_dirs],
            "all_capabilities": sorted({
                cap
                for m in self._manifests.values()
                for cap in m.capabilities
            }),
        }

    def __repr__(self) -> str:
        return f"PluginLoader(manifests={self.manifest_count}, loaded={self.loaded_count})"
