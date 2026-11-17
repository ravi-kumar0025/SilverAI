"""Read-only final evaluation-readiness audit; no training or scoring."""
from __future__ import annotations

import csv
import hashlib
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from urllib.request import urlopen

# Make both `python -m scripts.final_verify` and `python scripts/final_verify.py`
# resolve the repository's package imports identically.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.artifact_contract import load_versioned_artifact, runtime_versions
from src.models.spot_regression import load_regression_artifacts
from src.features.spot_features import SPOT_FEATURE_COLUMNS
from src.spot_config import SPOT_CONFIG

OUT = ROOT / "reports" / "spot_v3_20261115"
MANIFEST = ROOT / "data" / "dataset_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the read-only SilverAI evaluation-readiness audit.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="Backend base URL used for the optional live-status check.")
    parser.add_argument("--skip-live", action="store_true", help="Skip the live endpoint check and mark live evidence as unavailable.")
    return parser.parse_args()


def parse_figure_index(text: str) -> tuple[list[str], list[str]]:
    section = None
    generated: list[str] = []
    missing: list[str] = []
    for line in text.splitlines():
        if line == "Generated:":
            section = generated
        elif line == "MISSING:":
            section = missing
        elif line.startswith("- ") and section is not None:
            section.append(line[2:].strip())
    return generated, missing


def main() -> None:
    args = parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: object) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": str(detail)})

    check("dataset_id", manifest.get("dataset_id") == "4be8732dbab3aaa6", manifest.get("dataset_id"))
    check("holdout_status", SPOT_CONFIG.holdout_status == "used_do_not_reuse", SPOT_CONFIG.holdout_status)
    for filename, info in manifest.get("files", {}).items():
        path = ROOT / SPOT_CONFIG.data_root / filename
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        check(f"hash:{filename}", actual == info.get("sha256"), actual or "missing")

    classifier = load_versioned_artifact(SPOT_CONFIG.root_model / "spot_lightgbm.joblib", artifact_kind="spot_classifier")
    check("classifier_artifact", tuple(classifier.get("features", ())) == SPOT_FEATURE_COLUMNS, len(classifier.get("features", ())))
    regression = load_regression_artifacts(SPOT_CONFIG.root_regression_model)
    check("regression_artifacts", set(regression) == {"mean", "q10", "q50", "q90"}, sorted(regression))
    check("feature_count", len(classifier.get("features", ())) == 34, len(classifier.get("features", ())))
    check("regression_metrics", (ROOT / "reports" / "spot_v3_20261116_regression" / "regression_metrics.csv").is_file(), "stored CSV")
    figure_index = (OUT / "figures_index.md").read_text(encoding="utf-8")
    generated, missing_figures = parse_figure_index(figure_index)
    check("report_figures", "Generated:" in figure_index and "MISSING:" in figure_index, "figures_index.md")
    check("report_figures_complete", not missing_figures, missing_figures or "all indexed figures present")
    if args.skip_live:
        live = {"status": "skipped", "reason": "--skip-live"}
    else:
        try:
            with urlopen(f"{args.api_url.rstrip('/')}/api/live/status", timeout=3) as response:
                live = json.loads(response.read().decode("utf-8"))
            check("live_endpoint", response.status == 200, live)
            check("live_companions", not live.get("companion_missing"), live.get("companion_missing"))
            check("live_freshness_observed", bool(live.get("fresh")), live.get("age_seconds"))
        except Exception as exc:
            live = {"status": "unavailable", "error": str(exc)}
            check("live_endpoint", False, exc)

    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "api_url": args.api_url, "runtime_versions": runtime_versions(), "dataset_id": manifest.get("dataset_id"), "checks": checks, "live_observation": live}
    (OUT / "final_audit.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    with (OUT / "final_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "status", "detail"]); writer.writeheader(); writer.writerows(checks)
    passed = sum(row["status"] == "PASS" for row in checks)
    failed = sum(row["status"] == "FAIL" for row in checks)
    live_ready = bool(live.get("fresh")) and not live.get("companion_missing")
    summary = [
        "# SilverAI evaluation-readiness summary",
        "",
        f"Generated at `{payload['generated_at']}` by `{Path(__file__).name}`.",
        f"Dataset ID: `{manifest.get('dataset_id')}`",
        f"Checks: **{passed} passed, {failed} failed**",
        f"Runtime versions: `{json.dumps(runtime_versions(), sort_keys=True)}`",
        "",
        "## Readiness",
        "",
        f"- Static/model/report readiness: **{'PASS' if failed == 0 else 'REVIEW'}**",
        f"- Live-feed readiness at audit time: **{'PASS' if live_ready else 'STALE/REVIEW/UNAVAILABLE'}**",
        f"- Holdout policy: `{SPOT_CONFIG.holdout_status}`",
        "",
        "## Report assets",
        "",
        f"- Generated figures: {len(generated)}",
        f"- Missing figures: {len(missing)}",
    ]
    summary.extend(f"  - `{name}`" for name in missing)
    summary.extend([
        "",
        "## Live observation",
        "",
        f"- Provider: `{live.get('provider', 'unknown')}`",
        f"- Fresh: `{live.get('fresh', False)}`",
        f"- Age seconds: `{live.get('age_seconds')}`",
        f"- Companion missing: `{live.get('companion_missing', [])}`",
        "",
        "A stale live result is reported honestly and is not converted into a pass.",
    ])
    (OUT / "readiness_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "failed": failed, "generated_at": payload["generated_at"], "live": live}, default=str))


if __name__ == "__main__":
    main()
