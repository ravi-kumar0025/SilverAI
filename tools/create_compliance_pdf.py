from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path("output/pdf/silverai_compliance_matrix.pdf")
OUT.parent.mkdir(parents=True, exist_ok=True)
rows = [
    ("Declared asset", "COMPLETE", "COMEX Silver Futures / SI is consistent across data, backend, and frontend."),
    ("Historical five-minute data", "COMPLETE", "Validated, timezone-aware OHLCV loader and historical adapter."),
    ("Vendor and roll provenance", "INCOMPLETE", "Original CSV vendor and contract-roll policy are not recorded."),
    ("Chronological model validation", "COMPLETE", "Purged chronological train, validation, and untouched test partitions."),
    ("Trained artifacts", "COMPLETE", "Classifier, regressor, metadata, and selection artifacts are present."),
    ("Baseline comparison", "INCOMPLETE", "Majority/zero baselines exist; UP/DOWN/RSI baselines need report regeneration."),
    ("Probability calibration", "MISSING", "No calibrated probability model or calibration report is present."),
    ("Live COMEX provider", "MISSING", "No authenticated, tested provider adapter is configured."),
    ("Live candle orchestration", "MISSING", "Reconnect, session, close-confirmation, and stale-feed loop require a real provider."),
    ("Historical chart", "COMPLETE", "Validated candles, volume, range controls, zoom, pan, and reset are wired."),
    ("Prediction persistence", "INCOMPLETE", "SQLite journal, deduplication, confirmed-candle API, and outcome storage are present."),
    ("No repainting", "INCOMPLETE", "Append-only storage and tests exist; live orchestration is not connected."),
    ("Outcome evaluation", "INCOMPLETE", "Full-horizon evaluation is implemented; no live scheduler exists."),
    ("Paper trading", "INCOMPLETE", "Cost/slippage-aware evaluator exists; no live signal stream drives it."),
    ("Live validation UI", "INCOMPLETE", "Validation page and APIs exist; live markers and full metrics remain."),
    ("Five-day readiness", "NOT READY", "Blocked by live provider, calibration, orchestration, and live verification."),
]
status_colors = {"COMPLETE": "#dff2e8", "INCOMPLETE": "#fff0cc", "MISSING": "#fbdede", "NOT READY": "#f5cccc"}
with PdfPages(OUT) as pdf:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    fig.text(.5, .965, "SilverAI Compliance Matrix", ha="center", va="top", fontsize=18, weight="bold", color="#15232b")
    fig.text(.5, .935, "Evidence-backed audit | Final verdict: NOT READY FOR FIVE-DAY LIVE TESTING", ha="center", va="top", fontsize=8.5, color="#5b6c75")
    table = ax.table(cellText=[[a, s, e] for a, s, e in rows], colLabels=["Requirement area", "Status", "Evidence and remaining gap"], colWidths=[.22, .13, .65], cellLoc="left", bbox=[.03, .06, .94, .82])
    table.auto_set_font_size(False)
    table.set_fontsize(7.1)
    table.scale(1, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#c8d3d7")
        cell.set_linewidth(.4)
        if row == 0:
            cell.set_facecolor("#233942")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(status_colors.get(rows[row - 1][1], "white") if col == 1 else "white")
            cell.get_text().set_color("#23343c")
            if col == 1:
                cell.get_text().set_weight("bold")
                cell.get_text().set_ha("center")
    fig.text(.03, .025, "Historical mode is reported honestly when no verified live provider is configured. No fabricated data, confidence, or performance is used.", fontsize=7.5, color="#5b6c75")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
print(OUT)
