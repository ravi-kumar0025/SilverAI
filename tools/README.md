# Research and audit tools

The tools directory contains study runners and final checks for the active XAG/USD spot project.

| Tool | Purpose |
| --- | --- |
| `run_final_audit.py` | Read-only submission audit for canonical data, feature schema, required report artifacts, configuration, and stale SI/futures references. |
| `run_step0_study.py` | Runs the initial spot study stage. |
| `run_remaining_study.py` | Runs the remaining configured study variants. |
| `run_regression_threshold_study.py` | Runs the regression/threshold study. |
| `run_best_variant_extras.py` | Produces additional evidence for the selected variant. |
| `generate_spot_v3.py` | Generates the configured spot model/report artifacts. |

Use the read-only checks before submission:

```powershell
.\.venv\Scripts\python.exe tools/run_final_audit.py
.\.venv\Scripts\python.exe scripts/final_verify.py
```

Study runners can overwrite or regenerate report artifacts. Do not rerun selection against the used holdout; see [`docs/reproducibility.md`](../docs/reproducibility.md) and [`DECISIONS.md`](../DECISIONS.md).
