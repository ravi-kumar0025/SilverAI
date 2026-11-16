"""Build the SilverAI academic project report as a polished DOCX."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".report_deps"))
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "SilverAI_Project_Report.docx"
REPORT = ROOT / "reports" / "spot_v3_20261115"
FIGURES = REPORT / "figures"
TABLES = REPORT / "tables"

NAVY = "000000"
TEAL = "000000"
GOLD = "000000"
INK = "222222"
MUTED = "555555"
PALE = "FFFFFF"
LIGHT = "FFFFFF"
RED = "000000"


def shade(cell, color: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), color)


def set_cell_borders(cell, color="D7DDE3", size="6") -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = qn(f"w:{edge}")
        node = borders.find(tag)
        if node is None:
            node = OxmlElement(f"w:{edge}")
            borders.append(node)
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), size)
        node.set(qn("w:space"), "0")
        node.set(qn("w:color"), color)


def set_cell_text(cell, text: str, *, bold=False, color=INK, size=8.5) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run(str(text))
    run.bold = bold
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    run.font.name = "Aptos"
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_cell_margins(cell, top=80, start=90, bottom=80, end=90) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for side, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def add_table(doc, headers, rows, widths=None, compact=False):
    # Use aligned text blocks instead of Word table objects, matching the
    # plain academic presentation of the supplied reference report.
    header = doc.add_paragraph()
    header.paragraph_format.space_after = Pt(2)
    run = header.add_run("  |  ".join(str(value) for value in headers))
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(8 if compact else 9)
    run.font.color.rgb = RGBColor.from_string(INK)
    for row in rows:
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.left_indent = Inches(0.12)
        paragraph.paragraph_format.space_after = Pt(1.5)
        run = paragraph.add_run("  |  ".join(str(value) for value in row))
        run.font.name = "Times New Roman"
        run.font.size = Pt(7.5 if compact else 8.5)
        run.font.color.rgb = RGBColor.from_string(INK)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return None


def add_caption(doc, number, text):
    p = doc.add_paragraph(style="Caption")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(f"Figure {number}. {text}")
    run.font.color.rgb = RGBColor.from_string(MUTED)
    run.font.size = Pt(8.5)
    return p


def add_table_caption(doc, number, text):
    p = doc.add_paragraph(style="Caption")
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(f"{number}. {text}")
    run.font.color.rgb = RGBColor.from_string(MUTED)
    run.font.size = Pt(8.5)
    return p


def add_figure(doc, number, filename, caption, width=6.25):
    path = FIGURES / filename
    if not path.is_file():
        p = doc.add_paragraph(f"Figure {number} unavailable: {filename}")
        p.style = "Caption"
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    picture = p.add_run().add_picture(str(path), width=Inches(width))
    picture._inline.docPr.set("descr", caption)
    picture._inline.docPr.set("title", f"Figure {number}")
    add_caption(doc, number, caption)


def heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    p.paragraph_format.keep_with_next = True
    return p


def body(doc, text, *, first=True):
    p = doc.add_paragraph(style="Body Text")
    p.paragraph_format.first_line_indent = Inches(0.22) if first else Inches(0)
    p.add_run(text)
    return p


def bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.add_run(text)
    return p


def page_break(doc):
    doc.add_page_break()


def field_page_number(paragraph):
    run = paragraph.add_run()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    run._r.append(fld)


def setup_doc(doc):
    section = doc.sections[0]
    section.top_margin = Inches(0.9)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.9)
    section.right_margin = Inches(0.9)
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor.from_string(INK)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15
    for name, size, color in (("Title", 22, NAVY), ("Heading 1", 16, NAVY), ("Heading 2", 12, NAVY), ("Heading 3", 11, NAVY)):
        style = styles[name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(10 if name != "Title" else 0)
        style.paragraph_format.space_after = Pt(5)
        style.paragraph_format.keep_with_next = True
    styles["Caption"].font.name = "Times New Roman"
    styles["Caption"].font.size = Pt(9)
    styles["Caption"].font.italic = True
    styles["Caption"].font.color.rgb = RGBColor.from_string(MUTED)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run("SilverAI | Research report | ")
    field_page_number(footer)
    for run in footer.runs:
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor.from_string(MUTED)


def read_csv(name):
    with (TABLES / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(value, digits=3):
    if value in (None, "", "nan", "NaN"):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def build():
    metrics = json.loads((REPORT / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "data" / "dataset_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((REPORT / "final_audit.json").read_text(encoding="utf-8"))
    doc = Document()
    setup_doc(doc)

    # Title page follows the supplied academic report layout.
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Inches(0.35)
    r = p.add_run("MACHINE LEARNING PROJECT REPORT")
    r.bold = True; r.font.name = "Times New Roman"; r.font.size = Pt(13); r.font.color.rgb = RGBColor.from_string(NAVY)
    p = doc.add_paragraph(style="Title"); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("XAG USD Spot Market Intelligence")
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(30)
    r = p.add_run("A leakage-aware machine learning study with closed-candle live inference")
    r.font.name = "Times New Roman"; r.font.size = Pt(12); r.font.color.rgb = RGBColor.from_string(INK)
    for text, size, bold, after in [
        ("Project Report", 12, True, 25),
        ("Submitted by", 11, False, 8),
        ("Ravi Kumar", 12, True, 3),
        ("Roll No. 2401AI51", 11, False, 28),
        ("2026", 11, False, 0),
    ]:
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(after)
        r = p.add_run(text); r.font.name = "Times New Roman"; r.font.size = Pt(size); r.bold = bold; r.font.color.rgb = RGBColor.from_string(INK)
    page_break(doc)

    heading(doc, "Abstract", 1)
    body(doc, "SilverAI is a research terminal for XAG/USD spot that combines causal market features, purged walk-forward validation, calibrated classification, and one-hour return regression. The system is built around complete five-minute candles assembled from Dukascopy one-minute bid data for silver, gold, and EUR/USD. A canonical 34-feature schema describes silver price action, volatility, trend and momentum, cross-market relationships, and UTC session context. The deployed research path predicts the sign of the one-hour move for an abstaining Up/Down indicator and separately predicts the forward log return with a 10-90% conditional range.")
    body(doc, "The evidence is deliberately development-only. The used holdout from 2026-10-01 through 2026-11-13 is not reused in this report. The stored regression study does not beat the zero-return baseline consistently, and the classification results do not support a claim of a reliable trading edge. The appropriate conclusion is therefore a selective, paper-trading research indicator with explicit abstention, immutable forecasts, and strong data-quality gates rather than an automated trading system.")
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(8); p.paragraph_format.space_after = Pt(4)
    r = p.add_run("Keywords: "); r.bold = True; r.font.color.rgb = RGBColor.from_string(TEAL)
    p.add_run("XAG/USD, time-series validation, LightGBM, calibration, regression forecasting, live data quality, paper trading")

    heading(doc, "Contents", 1)
    for line in ["1 Introduction", "2 Literature Review", "3 Proposed Methodology", "4 Results and Discussion", "5 Conclusion and Future Scope", "References", "Appendix A Reproducibility and Readiness"]:
        p = doc.add_paragraph(line); p.paragraph_format.left_indent = Inches(0.18); p.paragraph_format.space_after = Pt(2)
    heading(doc, "List of Figures and Tables", 1)
    for line in ["Figure 3.1 SilverAI system architecture", "Figure 3.2 XAG/USD price history and used holdout", "Figure 3.3 Data coverage and gaps", "Figure 3.4 Label construction and exclusions", "Figure 3.5 Development class balance", "Figure 3.6 Purged walk-forward layout", "Figure 4.1 Per-fold accuracy", "Figure 4.2 Per-fold AUC", "Figure 4.3 Stored horizon and label comparison", "Figure 4.4 Confident-signal thresholds", "Figure 4.5 Cost sensitivity", "Figure 4.6 Seed stability", "Figure 4.7 Regression results", "3.1 Data and provenance summary", "4.1 Classification results by fold", "4.2 Regression results by fold", "4.3 Baselines and cost scenarios"]:
        p = doc.add_paragraph(line); p.paragraph_format.left_indent = Inches(0.18); p.paragraph_format.space_after = Pt(1)
    page_break(doc)

    # Chapter 1
    heading(doc, "1 Introduction", 1)
    heading(doc, "1.1 Background and Motivation", 2)
    body(doc, "Short-horizon market prediction is difficult for a structural reason: the target is small relative to noise, market observations are asynchronous, and the apparent sample size can be inflated by overlapping windows. A useful research system must therefore be judged as much by its data contract and temporal discipline as by its model score. SilverAI addresses this engineering problem for XAG/USD spot by making candle closure, companion-market availability, feature ordering, and forecast immutability explicit.")
    heading(doc, "1.2 Problem Statement", 2)
    body(doc, "The project asks whether a reproducible 34-feature pipeline can produce calibrated directional information and a one-hour return estimate from closed five-minute XAG/USD candles, while remaining honest about uncertainty, gaps, costs, and live-feed latency. The task is not to predict raw price levels or to claim profitability. The price forecast is derived from the modelled return as close_now times exp(predicted_return), and the interface must abstain when the evidence is weak.")
    heading(doc, "1.3 Objectives", 2)
    for x in ["Build a spot-only, UTC-aligned data contract with no invented volume.", "Generate causal features only after a candle is closed and preserve the ordered 34-feature schema.", "Compare raw and calibrated classification under purged walk-forward validation.", "Train return and quantile regressors without modelling price levels directly.", "Deliver an auditable historical/live terminal that reports stale data, missing companions, and immutable forecasts explicitly."]:
        bullet(doc, x)
    heading(doc, "1.4 Contributions", 2)
    body(doc, "The project contributions are practical rather than claims of a new trading theory. First, it provides a closed-candle alignment path for three Dukascopy instruments. Second, it enforces versioned artifact contracts with runtime-library checks. Third, it unifies historical, replay, and live-log inference guards. Fourth, it combines classification abstention with a return-regression range. Fifth, it preserves a forensic record of data identity, holdout use, costs, and known missing evidence.")
    page_break(doc)

    # Chapter 2
    heading(doc, "2 Literature Review", 1)
    heading(doc, "2.1 Existing Approaches", 2)
    body(doc, "Gradient-boosted decision trees remain a strong baseline for structured tabular data because they can represent nonlinear interactions without requiring a deep neural architecture. LightGBM uses histogram-based learning and leaf-wise growth to obtain efficient tree ensembles. Probability calibration is a separate problem: a classifier can rank examples well while producing probabilities that are too extreme or too conservative. Time-series validation adds another constraint because random shuffles allow overlapping future information to cross the train-test boundary.")
    body(doc, "Financial prediction studies commonly use technical indicators, cross-asset information, and regime descriptors. The risk is not the presence of such features but their timestamp semantics. A feature computed from a bar is only available when that bar closes, and a cross-market feature is only valid when the companion observation is known. SilverAI treats these conditions as part of the model specification rather than as implementation details.")
    heading(doc, "2.2 Research Gaps", 2)
    for x in ["Many reported scores are not directly comparable because labels, horizons, and costs differ.", "Random validation can overstate performance when forward-return windows overlap.", "Calibration and abstention are often reported separately from the live data-quality contract.", "Reduced error metrics do not establish trading utility when the target is close to a no-change forecast.", "A live interface can appear responsive while silently reusing forming candles or repainting old signals."]:
        bullet(doc, x)
    heading(doc, "2.3 Position of This Study", 2)
    body(doc, "SilverAI is positioned as an auditable engineering study. Its main result is not a claim that the model has discovered a persistent edge. Instead, it demonstrates a reproducible path from raw bid candles to immutable research forecasts and reports where the evidence is insufficient. The design follows the spirit of purged validation and calibration literature while keeping its claims narrower than a production trading system.")
    page_break(doc)

    # Chapter 3
    heading(doc, "3 Proposed Methodology", 1)
    heading(doc, "3.1 Dataset and Provenance", 2)
    body(doc, "The canonical manifest identifies dataset 4be8732dbab3aaa6. It contains XAG/USD, XAU/USD, and EUR/USD one-minute bid OHLC data from 2023-11-01 through 2026-11-13. The source has no volume field; the pipeline does not synthesize volume and no volume-derived feature is active. The original holdout is recorded as used and is not reused for the development evidence presented here.")
    add_table_caption(doc, "3.1", "Data and provenance used by the study.")
    add_table(doc, ["Item", "Value", "Interpretation"], [
        ["Dataset ID", manifest.get("dataset_id", "-"), "Canonical manifest identity"],
        ["XAG/USD rows", "1,334,781", "One-minute bid OHLC"],
        ["XAU/USD rows", "1,334,790", "Companion market"],
        ["EUR/USD rows", "1,342,487", "Companion market"],
        ["Research instrument", "XAGUSD_SPOT", "XAG/USD spot"],
        ["Primary timeframe", "5 minutes", "Complete left-labelled candles"],
        ["Forecast horizon", "12 bars / 60 minutes", "Return and directional target"],
        ["Used holdout", "2026-10-01 to 2026-11-13", "Do not reuse"],
    ], widths=[1.35, 1.7, 3.2])
    heading(doc, "3.2 Data Preprocessing and Feature Construction", 2)
    body(doc, "Raw rows are sorted and de-duplicated before complete candles are built. A five-minute candle requires all five one-minute observations. XAG/USD is never forward-filled; companion markets may be carried for at most two candles. The feature order is fixed by the source module and verified against every versioned artifact. Groups are silver price action, volatility, trend and momentum, cross-market, and time/session context.")
    heading(doc, "3.3 System Architecture", 2)
    add_figure(doc, "3.1", "fig_3_1_architecture.png", "Causal path from Dukascopy OHLC through closed-candle construction, features, labels, models, calibration, live bridge, and UI.")
    heading(doc, "3.4 Labels and Forecast Outputs", 2)
    body(doc, "The classification path uses an ATR-scaled dead zone: observations with a forward log return inside the configured band are not treated as directional evidence. The regression path instead models the timestamp-exact one-hour forward log return and its conditional quantiles. The deployed price estimate is a unit conversion from the return, not a raw price-level model. Forecasts are written once for a finalized candle and are never recomputed in place.")
    add_figure(doc, "3.2", "fig_3_4_label_construction.png", "The directional dead-zone and triple-barrier concepts used in development target studies.")
    heading(doc, "3.5 Live Bridge and Evaluation", 2)
    body(doc, "The owner-run Node helper polls Dukascopy real-time rates, writes raw one-minute JSONL events, and posts heartbeats/candles to the FastAPI service. The log-tail source discards forming candles, de-duplicates repeated events, builds complete five-minute bars, and exposes freshness and companion status. When the one-hour future window is complete and contiguous, the live cycle evaluates the persisted forecast; missing or gapped future windows remain pending rather than being guessed.")
    add_figure(doc, "3.3", "fig_3_6_walk_forward.png", "Stored purged walk-forward structure with train, purge, calibration, embargo, and validation blocks.")
    page_break(doc)

    # Chapter 4
    heading(doc, "4 Results and Discussion", 1)
    heading(doc, "4.1 Development Validation", 2)
    body(doc, "The development evidence uses chronological folds with interval purging and embargo. Raw and calibrated scores are reported together because calibration can change decision quality even when ranking performance remains similar. The rows below come from the stored fold table and are not a new evaluation run.")
    folds = read_csv("results_per_fold.csv")
    add_table_caption(doc, "4.1", "Classification metrics by development fold.")
    add_table(doc, ["Fold", "Rows", "Raw acc.", "Cal. acc.", "Raw AUC", "Cal. AUC", "Raw log loss", "Cal. log loss"], [[r.get("fold"), r.get("rows"), fmt(r.get("raw_accuracy"), 3), fmt(r.get("calibrated_accuracy"), 3), fmt(r.get("raw_auc"), 3), fmt(r.get("calibrated_auc"), 3), fmt(r.get("raw_log_loss"), 3), fmt(r.get("calibrated_log_loss"), 3)] for r in folds], compact=True)
    add_figure(doc, "4.1", "fig_4_1_accuracy.png", "Raw and calibrated accuracy across the stored development folds.")
    add_figure(doc, "4.2", "fig_4_2_auc_curves.png", "Raw and calibrated ROC AUC across the stored development folds.")
    heading(doc, "4.2 Return Regression", 2)
    body(doc, "The regression family predicts the forward log return at a one-hour horizon and separately estimates the 10th, 50th, and 90th conditional quantiles. The stored results are negative in R2 versus the zero-return forecast on every fold. That is an important negative result: the model can produce a numerically plausible forecast in the UI, but the current evidence does not establish improvement over no change.")
    reg = read_csv("regression_metrics.csv")
    add_table_caption(doc, "4.2", "Return-regression results on development folds.")
    add_table(doc, ["Fold", "Rows", "R2 vs zero", "MAE", "RMSE", "Spearman IC", "Sign acc.", "10-90 coverage"], [[r.get("fold"), r.get("rows"), fmt(r.get("r2_vs_zero"), 3), fmt(r.get("mae"), 4), fmt(r.get("rmse"), 4), fmt(r.get("spearman_ic"), 3), fmt(r.get("sign_accuracy"), 3), fmt(r.get("band_coverage"), 3)] for r in reg], compact=True)
    add_figure(doc, "4.3", "fig_4_9_regression_results.png", "Stored R2, sign accuracy, and 10-90 interval coverage for the regression family.")
    heading(doc, "4.3 Baselines, Selectivity, and Costs", 2)
    base = read_csv("baselines.csv")
    add_table_caption(doc, "4.3", "Baseline and selective-signal comparison.")
    add_table(doc, ["Variant", "Rows", "Accuracy", "Coverage", "Notes"], [[r.get("baseline", r.get("variant", "-")), r.get("rows", "-"), fmt(r.get("accuracy"), 3), fmt(r.get("coverage"), 3), r.get("notes", "same live-style rows")] for r in base[:8]], compact=True)
    costs = read_csv("costs.csv")
    add_table_caption(doc, "4.4", "Sensitivity of the stored result to transaction-cost assumptions.")
    add_table(doc, ["Cost multiple", "Signals", "Coverage", "Net return", "Variant"], [[r.get("cost_multiple"), r.get("signals", "-"), fmt(r.get("coverage"), 3), fmt(r.get("net_return"), 5), r.get("variant", "best stored variant")] for r in costs], compact=True)
    add_figure(doc, "4.4", "fig_4_4_thresholds.png", "Stored confident-signal accuracy across validation thresholds.")
    add_figure(doc, "4.5", "fig_4_7_profit_costs.png", "Stored cost sensitivity for the selected development variant.")
    add_figure(doc, "4.6", "fig_4_8_seed_stability.png", "Stored seed-stability evidence for the selected development variant.")
    heading(doc, "4.4 Discussion of Findings", 2)
    body(doc, "The results support a cautious interpretation. The classifier is useful as a structured research signal because it exposes calibrated confidence and abstains; the regression model supplies a directional estimate and range for context. Neither result should be described as a consistent trading edge. Calibration can reduce raw accuracy at the same AUC because calibration changes probability values and therefore the thresholded decision rule; ranking quality and probability reliability are different properties. The live bridge is designed to preserve that distinction rather than turn low-confidence output into a trade.")
    page_break(doc)

    # Chapter 5
    heading(doc, "5 Conclusion and Future Scope", 1)
    heading(doc, "5.1 Key Findings", 2)
    for x in ["A canonical, hash-verified Dukascopy spot dataset and 34-feature contract are in place.", "The used holdout is protected and is not reused by the current development evidence.", "Purged walk-forward validation and artifact runtime checks reduce common leakage and environment risks.", "The regression family predicts returns rather than raw prices, but its stored R2 values do not beat the zero-return baseline consistently.", "The live bridge now ingests closed candles, exposes freshness and companion status, persists forecasts immutably, and evaluates eligible outcomes automatically.", "The correct operating mode is research and paper trading, with explicit abstention and no profitability claim."]:
        bullet(doc, x)
    heading(doc, "5.2 Limitations", 2)
    body(doc, "The corpus begins in 2023 rather than 2019, so historical conclusions are limited to the verified manifest. Live data quality depends on the owner-run Dukascopy helper and can become stale. Several future windows contain gaps and correctly remain pending. The final report asset index still identifies two unavailable source figures: a feature-group correlation heatmap and a reliability plot. They are not fabricated. The current model is not promoted and should not be used for autonomous orders.")
    heading(doc, "5.3 Future Research Directions", 2)
    for x in ["Collect a longer, independently confirmed history and freeze a new unused holdout before any further tuning.", "Store probability-bin calibration records and generate a true reliability diagram from a separate calibration block.", "Add a reproducible feature-correlation artifact with its exact development sampling rule.", "Compare abstention policies under costs and session regimes without selecting thresholds on the final holdout.", "Run a pre-registered paper-trading observation period with provider latency, missing companion, and forecast-settlement logs."]:
        bullet(doc, x)

    heading(doc, "References", 1)
    refs = [
        "[1] Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS.",
        "[2] López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.",
        "[3] Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Advances in Large Margin Classifiers.",
        "[4] Niculescu-Mizil, A. and Caruana, R. (2005). Predicting good probabilities with supervised learning. ICML.",
        "[5] Breiman, L. (2001). Random Forests. Machine Learning, 45, 5-32.",
        "[6] SilverAI project artifacts: config.yaml, dataset_manifest.json, spot_v3_20261115 metrics and tables, and versioned regression artifacts. Accessed 16 November 2026.",
    ]
    for ref in refs: body(doc, ref, first=False)

    page_break(doc)
    heading(doc, "Appendix A Reproducibility and Readiness", 1)
    heading(doc, "A.1 Commands", 2)
    for cmd in [".\\.venv\\Scripts\\python.exe -m uvicorn backend.server:app --host 127.0.0.1 --port 8000", "node scripts/live_feed_logger.mjs", "cd frontend; npm run dev", ".\\.venv\\Scripts\\python.exe -m pytest -q", "npm test -- --run", "npm run build", ".\\.venv\\Scripts\\python.exe scripts\\verify_market_data_alignment.py", ".\\.venv\\Scripts\\python.exe scripts\\verify_project_integrity.py"]:
        p = doc.add_paragraph(style="No Spacing"); p.paragraph_format.left_indent = Inches(0.25); p.add_run(cmd).font.name = "Consolas"
    heading(doc, "A.2 Readiness Record", 2)
    checks = audit.get("checks", [])
    add_table(doc, ["Area", "Status", "Evidence"], [[c.get("check", "-"), c.get("status", "-"), str(c.get("detail", "-"))[:95]] for c in checks], compact=True)
    body(doc, "At the time of report preparation, the Python suite had 54 passing tests with five warnings, the frontend suite had four passing tests, and the production frontend build completed successfully. The readiness audit passed the data identity, artifact contract, regression artifact, feature schema, holdout policy, and live endpoint checks. The report-asset completeness check remains intentionally open for the two missing source-backed figures listed above.", first=False)

    doc.core_properties.title = "SilverAI XAG USD Spot Market Intelligence"
    doc.core_properties.subject = "Machine learning project report"
    doc.core_properties.author = "SilverAI Project Team"
    doc.core_properties.keywords = "XAG/USD, LightGBM, time series, forecasting, machine learning"
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
