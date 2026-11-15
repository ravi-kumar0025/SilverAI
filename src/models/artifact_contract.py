"""Runtime-version contract for serialized spot artifacts."""

from __future__ import annotations

import sys
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib


PACKAGE_NAMES = ("joblib", "numpy", "pandas", "scikit-learn", "lightgbm")


def runtime_versions() -> dict[str, str]:
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        **{name: version(name) for name in PACKAGE_NAMES},
    }


def version_mismatches(recorded: dict[str, str], current: dict[str, str] | None = None) -> dict[str, tuple[str | None, str]]:
    current = current or runtime_versions()
    return {key: (recorded.get(key), value) for key, value in current.items() if recorded.get(key) != value}


def dump_versioned_artifact(payload: Any, path: str | Path, *, artifact_kind: str) -> None:
    joblib.dump({"artifact_kind": artifact_kind, "runtime_versions": runtime_versions(), "payload": payload}, path)


def load_versioned_artifact(path: str | Path, *, artifact_kind: str) -> Any:
    path = Path(path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        envelope = joblib.load(path)
    if any(warning.category.__name__ == "InconsistentVersionWarning" for warning in caught):
        raise RuntimeError(f"Artifact {path} was serialized by an incompatible scikit-learn runtime.")
    if not isinstance(envelope, dict) or envelope.get("artifact_kind") != artifact_kind:
        raise RuntimeError(f"Artifact {path} is missing the {artifact_kind} contract.")
    recorded = envelope.get("runtime_versions")
    if not isinstance(recorded, dict):
        raise RuntimeError(f"Artifact {path} has no runtime version metadata.")
    mismatches = version_mismatches(recorded)
    if mismatches:
        detail = ", ".join(f"{key}: recorded={old!r}, running={new!r}" for key, (old, new) in mismatches.items())
        raise RuntimeError(f"Artifact {path} runtime mismatch: {detail}")
    return envelope["payload"]


__all__ = ["dump_versioned_artifact", "load_versioned_artifact", "runtime_versions", "version_mismatches"]
