"""Fail-fast alignment audit for the active XAG/USD spot submission."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features.market_features import SPOT_FEATURE_COLUMNS  # noqa: E402


def main() -> int:
    manifest_path = ROOT / "data" / "dataset_manifest.json"
    metrics_path = ROOT / "reports" / "spot_dukascopy_20261115" / "metrics.json"
    report_path = ROOT / "reports" / "spot_dukascopy_20261115" / "report.md"
    checks: list[tuple[str, bool, str]] = []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    checks.append(("manifest exists", bool(manifest), str(manifest_path)))
    checks.append(("manifest dataset_id matches metrics", bool(manifest.get("dataset_id") and manifest.get("dataset_id") == metrics.get("dataset_id")), str(metrics_path)))
    checks.append(("spot identity", config.get("instrument") == "XAGUSD_SPOT" and config.get("symbol") == "XAG/USD", "config.yaml"))
    checks.append(("Dukascopy bid UTC", config.get("data_source") == "Dukascopy" and config.get("price_side") == "bid" and config.get("timezone") == "UTC", "config.yaml"))
    checks.append(("34-feature contract", len(SPOT_FEATURE_COLUMNS) == 34 and metrics.get("feature_count") == 34, "src/features/market_features.py"))
    checks.append(("no volume-derived feature names", not any("volume" in feature.lower() for feature in SPOT_FEATURE_COLUMNS), str(SPOT_FEATURE_COLUMNS)))
    checks.append(("report has metadata id", report_path.is_file() and f"dataset_id: {manifest.get('dataset_id')}" in report_path.read_text(encoding="utf-8"), str(report_path)))
    forbidden = ("COMEX", "SI_1m", "silver_5m", "model_comparison", "roll-adjusted", "America/New_York", "17-feature", "three-class", "relative_volume", "volume_change")
    hits: list[str] = []
    scan_roots = (ROOT / "src", ROOT / "backend", ROOT / "frontend", ROOT / "tests", ROOT / "docs", ROOT / "tools", ROOT / "README.md", ROOT / "config.yaml", ROOT / ".env.example", ROOT / "DECISIONS.md")
    for root in scan_roots:
        paths = [root] if root.is_file() else root.rglob("*") if root.exists() else []
        for path in paths:
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for term in forbidden:
                if term.lower() in text.lower():
                    hits.append(f"{path.relative_to(ROOT)}:{term}")
    checks.append(("no forbidden active references", not hits, ", ".join(hits[:8])))
    failed = False
    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL':4} | {name:38} | {detail}")
        failed |= not passed
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
