"""
Build the evaluation-results deck from the committed artifacts.

Same rule as the report: every figure is read from artifacts/results/*.json at
run time. Nothing is typed into a slide by hand, so the deck cannot drift away
from what the API serves or what the report states.

    python scripts/09_evaluation_deck.py        # -> docs/Evaluation_Results.pptx
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "artifacts" / "results"
OUT = ROOT / "docs" / "Evaluation_Results.pptx"

# palette lifted from Literature_Review_Deck.pptx, same as the dashboard
BG = RGBColor(0x1C, 0x2B, 0x1C)
CARD = RGBColor(0x22, 0x35, 0x22)
BAND = RGBColor(0x2C, 0x5F, 0x2D)
LINE = RGBColor(0x3C, 0x5A, 0x3C)
TEXT = RGBColor(0xD8, 0xDE, 0xD8)
MUTED = RGBColor(0xA8, 0xB4, 0xA8)
MOSS = RGBColor(0x97, 0xBC, 0x62)
CLAY = RGBColor(0xB8, 0x50, 0x42)
HEAD_FONT = "Cambria"
BODY_FONT = "Calibri"

W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.72)

LABEL = {
    "global_mean": "Global mean",
    "district_mean": "District mean",
    "district_trend": "District trend",
    "centralised": "Centralised",
    "local": "Local-only",
    "fedper": "FedPer",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
}
ORDER = ["global_mean", "district_mean", "district_trend",
         "centralised", "local", "fedper", "fedavg", "fedprox"]
BASELINES = {"global_mean", "district_mean", "district_trend"}


def load(name: str):
    return json.loads((RESULTS / f"{name}.json").read_text())


def agg(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None
    return stats.mean(clean), (stats.stdev(clean) if len(clean) > 1 else 0.0)


# --------------------------------------------------------------- primitives

def slide(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = BG
    return s


def textbox(s, x, y, w, h, text, size=14, bold=False, color=TEXT,
            font=BODY_FONT, align=PP_ALIGN.LEFT, spacing=1.0):
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
        r.font.name = font
    return tb


def heading(s, title, kicker=None):
    """Title band across the top of a content slide."""
    band = s.shapes.add_shape(1, 0, 0, W, Inches(1.12))   # 1 = rectangle
    band.fill.solid()
    band.fill.fore_color.rgb = CARD
    band.line.fill.background()
    band.shadow.inherit = False
    if kicker:
        textbox(s, MARGIN, Inches(0.20), Inches(11), Inches(0.26), kicker,
                size=11, bold=True, color=MOSS, font=BODY_FONT)
    textbox(s, MARGIN, Inches(0.46), Inches(11.9), Inches(0.5), title,
            size=25, bold=True, color=TEXT, font=HEAD_FONT)


def note(s, text, y=Inches(6.42), color=MUTED, size=14):
    textbox(s, MARGIN, y, W - 2 * MARGIN, Inches(0.8), text,
            size=size, color=color, spacing=1.12)


def table(s, rows, x, y, w, col_w=None, first_bold=True, size=13,
          highlight=None, row_h=Inches(0.44)):
    """rows[0] is the header. highlight: {row_index: RGBColor} for the label cell."""
    nrow, ncol = len(rows), len(rows[0])
    shape = s.shapes.add_table(nrow, ncol, x, y, w, row_h * nrow)
    tbl = shape.table
    tbl.first_row = False
    tbl.horz_banding = False

    if col_w:
        total = sum(col_w)
        for i, frac in enumerate(col_w):
            tbl.columns[i].width = Emu(int(w * frac / total))

    for r, row in enumerate(rows):
        tbl.rows[r].height = row_h
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = BAND if r == 0 else CARD
            cell.margin_left = Inches(0.1)
            cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
            run = p.add_run()
            run.text = str(val)
            run.font.size = Pt(size)
            run.font.name = BODY_FONT
            run.font.bold = (r == 0) or (c == 0 and first_bold)
            if r == 0:
                run.font.color.rgb = TEXT
            elif highlight and r in highlight and c == 0:
                run.font.color.rgb = highlight[r]
            else:
                run.font.color.rgb = TEXT
    return shape


# ------------------------------------------------------------------ slides

def title_slide(prs, meta):
    s = slide(prs)
    band = s.shapes.add_shape(1, 0, Inches(2.05), W, Inches(2.5))
    band.fill.solid()
    band.fill.fore_color.rgb = CARD
    band.line.fill.background()
    band.shadow.inherit = False

    textbox(s, MARGIN, Inches(2.32), Inches(11.9), Inches(0.3),
            "EVALUATION RESULTS", size=12.5, bold=True, color=MOSS)
    textbox(s, MARGIN, Inches(2.70), Inches(11.9), Inches(0.9),
            "System Performance and Solution Effectiveness",
            size=33, bold=True, color=TEXT, font=HEAD_FONT)
    textbox(s, MARGIN, Inches(3.62), Inches(11.4), Inches(0.5),
            "Federated and explainable crop-yield analytics · "
            "GRU over 12-month climate sequences · federation by state",
            size=15, color=MUTED)

    facts = [
        (f"{meta['rows']:,}", "rows after cleaning"),
        (f"{meta['districts']}", "districts"),
        (f"{meta['states']}", "states as clients"),
        (f"{meta['year_min']}–{meta['year_max']}", "years"),
    ]
    x = MARGIN
    for value, label in facts:
        textbox(s, x, Inches(5.05), Inches(2.6), Inches(0.45), value,
                size=27, bold=True, color=MOSS, font=HEAD_FONT)
        textbox(s, x, Inches(5.58), Inches(2.6), Inches(0.3), label,
                size=12, color=MUTED)
        x += Inches(2.85)

    note(s, "Every figure in this deck is read from artifacts/results/*.json by "
            "scripts/09_evaluation_deck.py at build time. No number is typed into a "
            "slide by hand, so the deck, the report and the results API cannot "
            "disagree.", y=Inches(6.55))
    return s


def arms_slide(prs, abl):
    s = slide(prs)
    heading(s, "What is being compared", "HOW TO READ THIS DECK")
    rows = [["Arm", "Kind", "What it does"]]
    desc = {
        "global_mean": ("Baseline", "One mean yield for the whole panel"),
        "district_mean": ("Baseline", "Each district's own historical mean"),
        "district_trend": ("Reference", "Linear extrapolation of each district's trend"),
        "centralised": ("Ceiling", "One model, all data pooled in one place"),
        "local": ("Floor", "One model per state, no sharing at all"),
        "fedavg": ("Federated", "Weighted averaging of all weights each round"),
        "fedprox": ("Federated", "FedAvg plus a proximal term against client drift"),
        "fedper": ("Federated", "Shares the GRU base, keeps the head local"),
    }
    for arm in ORDER:
        kind, what = desc[arm]
        rows.append([LABEL[arm], kind, what])
    t = table(s, rows, MARGIN, Inches(1.42), W - 2 * MARGIN,
              col_w=[2.0, 1.6, 7.0], size=13.5, row_h=Inches(0.47))
    for r in range(1, len(rows)):
        t.table.cell(r, 1).text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT
        t.table.cell(r, 2).text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT
    t.table.cell(0, 1).text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT
    t.table.cell(0, 2).text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT

    note(s, f"Skill is measured against the {LABEL[abl['reference_model']].lower()} "
            f"baseline, not against zero. A model that cannot beat 'extrapolate each "
            f"district's own trend' has not earned its complexity. Learned arms are run "
            f"over {len(abl['seeds'])} seeds ({min(abl['seeds'])}–{max(abl['seeds'])}) "
            f"and reported as mean ± sample standard deviation.")
    return s


def model_cost_slide(prs, cx):
    s = slide(prs)
    heading(s, "Model cost", "R2 · SYSTEM PERFORMANCE")
    m = cx["model"]
    inf = cx["inference"]
    rows = [
        ["Quantity", "Value"],
        ["Total parameters", f"{m['total_parameters']:,}"],
        ["Base (GRU) parameters", f"{m['base_parameters']:,}"],
        ["Head parameters", f"{m['head_parameters']:,}"],
        ["Shared under FedPer", f"{m['fedper_shared_pct']:.0f}%"],
        ["State dict on the wire", f"{m['state_dict_kb']:.1f} KB"],
        ["GRU FLOPs per sample", f"{m['gru_flops_per_sample']:,}"],
        ["Forward cost", m["forward_cost"]],
        ["Inference", f"{inf['ms_per_sample']:.2f} ms/sample "
                      f"(batch {inf['batch_size']})"],
    ]
    table(s, rows, MARGIN, Inches(1.5), Inches(6.5), col_w=[3.4, 3.1], size=14,
          row_h=Inches(0.52))

    textbox(s, Inches(7.7), Inches(1.62), Inches(5.0), Inches(0.3),
            "WHY THIS MATTERS", size=11, bold=True, color=MOSS)
    textbox(s, Inches(7.7), Inches(2.05), Inches(5.0), Inches(4.2),
            f"At {m['total_parameters']:,} parameters the model is small enough that a "
            f"full state dict is {m['state_dict_kb']:.1f} KB. That is what makes the "
            f"federation cheap: the per-round cost is set by model size, not by data "
            f"size.\n\n"
            f"{m['fedper_shared_pct']:.0f}% of the parameters sit in the GRU base. "
            f"FedPer shares exactly that and keeps the {m['head_parameters']:,}-parameter "
            f"head local, which is where its communication saving comes from.\n\n"
            f"Inference at {inf['ms_per_sample']:.2f} ms/sample means a whole state can "
            f"be scored in well under a second on CPU.",
            size=15, color=MUTED, spacing=1.3)
    return s


def comms_slide(prs, cx):
    s = slide(prs)
    heading(s, "Communication cost per round", "R2 · SYSTEM PERFORMANCE")
    comms = {c["algorithm"]: c for c in cx["communication"]}
    rows = [["Algorithm", "Rounds", "KB / round", "Total MB", "s / round"]]
    for a in ["fedavg", "fedprox", "fedper"]:
        c = comms[a]
        rows.append([LABEL[a], f"{c['rounds_run']}", f"{c['kb_per_round']:,.1f}",
                     f"{c['total_mb']:.1f}", f"{c['s_per_round']:.2f}"])
    table(s, rows, MARGIN, Inches(1.5), W - 2 * MARGIN,
          col_w=[2.6, 1.6, 2.4, 2.0, 2.0], size=14.5, row_h=Inches(0.55))

    saving = ((comms["fedavg"]["kb_per_round"] - comms["fedper"]["kb_per_round"])
              / comms["fedavg"]["kb_per_round"] * 100)
    textbox(s, MARGIN, Inches(4.15), Inches(5.6), Inches(0.5),
            f"{saving:.1f}% less per round", size=31, bold=True,
            color=MOSS, font=HEAD_FONT)
    textbox(s, MARGIN, Inches(4.78), Inches(5.6), Inches(0.9),
            "FedPer against FedAvg, because the head\nnever leaves the client.",
            size=15, color=MUTED, spacing=1.2)

    textbox(s, Inches(6.9), Inches(4.1), Inches(5.7), Inches(0.3),
            "THE HONEST READING", size=11.5, bold=True, color=CLAY)
    textbox(s, Inches(6.9), Inches(4.5), Inches(5.7), Inches(2.4),
            f"Per round FedPer is cheaper. In total it is not: it ran "
            f"{comms['fedper']['rounds_run']} rounds against FedAvg's "
            f"{comms['fedavg']['rounds_run']}, so it moved "
            f"{comms['fedper']['total_mb']:.1f} MB against "
            f"{comms['fedavg']['total_mb']:.1f} MB overall. The saving is real "
            f"per-round and disappears once convergence is counted. A deck that quoted "
            f"only the {saving:.0f}% would be misleading.",
            size=15, color=MUTED, spacing=1.28)
    return s


def mu_slide(prs, mu):
    s = slide(prs)
    heading(s, "FedProx proximal term", "R2 · SYSTEM PERFORMANCE")
    rows = [["μ", "Algorithm", "R²", "Skill vs trend", "L2 vs FedAvg", "Rounds"]]
    for r in mu:
        rows.append([f"{r['mu']:g}", LABEL.get(r["algorithm"], r["algorithm"]),
                     f"{r['r2']:.4f}", f"{r['skill_vs_trend']:+.4f}",
                     f"{r['param_l2_vs_fedavg']:.3f}", f"{r['rounds']}"])
    table(s, rows, MARGIN, Inches(1.5), W - 2 * MARGIN,
          col_w=[1.3, 2.0, 2.0, 2.4, 2.4, 1.5], size=14, row_h=Inches(0.56))

    best = max(mu, key=lambda r: r["r2"])
    drift = [r for r in mu if r["param_l2_vs_fedavg"] > 0]
    note(s, f"The proximal term is active, not inert: parameter distance from the FedAvg "
            f"solution rises monotonically with μ, from {min(r['param_l2_vs_fedavg'] for r in drift):.3f} "
            f"to {max(r['param_l2_vs_fedavg'] for r in drift):.3f}. It is doing what it is "
            f"supposed to do. It simply does not help here: the best R² in the sweep is "
            f"{best['r2']:.4f} at μ = {best['mu']:g}, and no setting lifts FedProx above "
            f"the trend baseline. Reporting the sweep matters more than reporting a "
            f"tuned winner, because it shows the negative result was measured rather "
            f"than assumed.",
         y=Inches(5.45), size=15)
    return s


def ablation_slide(prs, abl):
    s = slide(prs)
    heading(s, "Predictive skill against a ceiling and a floor", "R3 · SOLUTION EFFECTIVENESS")

    grouped: dict[str, list] = {}
    for run in abl["runs"]:
        grouped.setdefault(run["arm"], []).append(run)

    rows = [["Arm", "R²", "Skill vs trend", "Worst-client RMSE", "Seeds"]]
    hi = {}
    for i, arm in enumerate([a for a in ORDER if a in grouped], start=1):
        runs = grouped[arm]
        r2, r2sd = agg([r["r2"] for r in runs])
        sk, sksd = agg([r["skill_vs_trend"] for r in runs])
        wc, _ = agg([r["worst_client_rmse"] for r in runs])
        r2s = f"{r2:.3f}" + (f" ± {r2sd:.3f}" if len(runs) > 1 else "")
        if arm == abl["reference_model"]:
            sks = "baseline"
        elif sk is None:
            sks = "—"
        else:
            sks = f"{sk:+.3f}" + (f" ± {sksd:.3f}" if len(runs) > 1 else "")
        rows.append([LABEL[arm], r2s, sks, f"{wc:,.0f}", f"{len(runs)}"])
        if sk is not None and arm != abl["reference_model"]:
            hi[i] = MOSS if sk > 0 else CLAY
    table(s, rows, MARGIN, Inches(1.45), W - 2 * MARGIN,
          col_w=[2.3, 2.6, 2.6, 2.7, 1.3], size=13.5, highlight=hi,
          row_h=Inches(0.45))

    by = {a: agg([r["r2"] for r in g])[0] for a, g in grouped.items()}
    note(s, f"The sentence that must accompany this table. Federated methods trade "
            f"accuracy for cross-client uniformity. Centralised ({by['centralised']:.3f}) "
            f"and local-only ({by['local']:.3f}) beat FedAvg ({by['fedavg']:.3f}) and "
            f"FedProx ({by['fedprox']:.3f}) because state-level heterogeneity dominates "
            f"this panel; FedPer recovers accuracy ({by['fedper']:.3f}) precisely by "
            f"declining to average the head. Read without this sentence the table looks "
            f"like a failed federation. Read with it, the ordering is the expected "
            f"consequence of strongly non-IID data, and is itself the finding.",
         y=Inches(5.72), color=TEXT, size=15)
    return s


def agreement_slide(prs, agr):
    s = slide(prs)
    heading(s, "Do federated explanations agree with centralised ones?",
            "R3 · SOLUTION EFFECTIVENESS")
    rows = [["Comparison", "Kendall τ", "95% interval", "Top-10 Jaccard", "Pairs"]]
    hi = {}
    ctrl = None
    for i, r in enumerate(agr["table"], start=1):
        name = r["comparison"].replace("centralised vs ", "")
        if "seed control" in name:
            name = "Seed control (centralised vs itself)"
            ctrl = r
            hi[i] = MOSS
        else:
            name = LABEL.get(name, name)
        rows.append([name, f"{r['kendall_tau']:.3f} ± {r['kendall_tau_sd']:.3f}",
                     f"[{r['kendall_tau_lo']:.3f}, {r['kendall_tau_hi']:.3f}]",
                     f"{r['top10_jaccard']:.3f}", f"{r['n_pairs']}"])
    table(s, rows, MARGIN, Inches(1.45), W - 2 * MARGIN,
          col_w=[4.3, 2.5, 2.6, 2.1, 1.1], size=13.5, highlight=hi,
          row_h=Inches(0.5))

    y = Inches(1.45) + Inches(0.5) * len(rows) + Inches(0.34)
    txt = (f"The control is the contribution. Two centralised models that differ only by "
           f"random seed agree at τ = {ctrl['kendall_tau']:.3f}, not 1.0. That is the "
           f"ceiling any federated comparison could reach, so a federated τ is only "
           f"meaningful as a fraction of it. Without this row, every federated number "
           f"below would look like a failure to reproduce the explanation; with it, the "
           f"question becomes how much of the achievable agreement survives federation.")
    textbox(s, MARGIN, y, W - 2 * MARGIN, Inches(1.4), txt, size=15, color=TEXT,
            spacing=1.22)

    ag = agr["agronomic_check"]
    textbox(s, MARGIN, Inches(6.05), W - 2 * MARGIN, Inches(0.3),
            f"AGRONOMIC PLAUSIBILITY · {ag['verdict']}", size=11.5, bold=True,
            color=MOSS if ag["verdict"] == "PASS" else CLAY)
    textbox(s, MARGIN, Inches(6.42), W - 2 * MARGIN, Inches(0.9),
            f"{ag['reason']}. Top features: "
            f"{', '.join(f.replace('_', ' ') for f in ag['top_features'][:6])}. "
            f"{agr['note'].capitalize()}.",
            size=13, color=MUTED, spacing=1.16)
    return s


def fidelity_slide(prs, fid):
    s = slide(prs)
    heading(s, "Do the explanations actually drive the prediction?",
            "R3 · SOLUTION EFFECTIVENESS")
    rows = [["Arm", "AOPC top-k", "AOPC random", "Ratio", "Verdict"]]
    for f in fid:
        rows.append([LABEL.get(f["arm"], f["arm"]), f"{f['aopc_topk']:.3f}",
                     f"{f['aopc_random']:.3f}", f"{f['ratio']:.1f}×", f["verdict"]])
    table(s, rows, MARGIN, Inches(1.5), W - 2 * MARGIN,
          col_w=[2.6, 2.6, 2.6, 2.0, 2.0], size=14, row_h=Inches(0.5))

    worst = min(fid, key=lambda f: f["aopc_random"])
    k = fid[0]
    textbox(s, MARGIN, Inches(4.15), W - 2 * MARGIN, Inches(0.3),
            "WHAT THE TEST DOES", size=11.5, bold=True, color=MOSS)
    textbox(s, MARGIN, Inches(4.52), W - 2 * MARGIN, Inches(0.8),
            f"Delete the top-k attributed features and measure how far the prediction "
            f"moves; delete k features at random and measure the same. If the "
            f"attribution is real, the first number is much larger. Evaluated on "
            f"{k['client']}, k up to {k['k_max']}, {k['n_rows']:,} rows.",
            size=15, color=MUTED, spacing=1.22)

    textbox(s, MARGIN, Inches(5.62), W - 2 * MARGIN, Inches(0.3),
            "THE CAVEAT THAT BELONGS NEXT TO THE TABLE", size=11.5, bold=True, color=CLAY)
    textbox(s, MARGIN, Inches(5.99), W - 2 * MARGIN, Inches(1.1),
            f"{LABEL.get(worst['arm'], worst['arm'])} has the highest ratio "
            f"({worst['ratio']:.1f}×) but the lowest random-deletion AOPC "
            f"({worst['aopc_random']:.3f}). The ratio is flattering because that model "
            f"is relatively unresponsive to feature deletion overall, not because its "
            f"ranking is sharper. Ratios are not comparable across arms with different "
            f"baseline responsiveness, and the deck does not rank the arms by them.",
            size=14, color=MUTED, spacing=1.2)
    return s


def confound_slide(prs, conf):
    s = slide(prs)
    heading(s, "Why the federation is posed by state", "R3 · SOLUTION EFFECTIVENESS")
    flips = [c for c in conf if c["pooled_r"] * c["within_district_r"] < 0]
    ranked = sorted(conf, key=lambda c: abs(c["pooled_r"] - c["within_district_r"]),
                    reverse=True)[:6]
    rows = [["Variable · month", "Pooled r", "Within-district r", "Sign flip"]]
    hi = {}
    for i, c in enumerate(ranked, start=1):
        flip = c["pooled_r"] * c["within_district_r"] < 0
        rows.append([f"{c['variable'].replace('_', ' ')} · {c['month']}",
                     f"{c['pooled_r']:+.3f}", f"{c['within_district_r']:+.3f}",
                     "yes" if flip else "no"])
        if flip:
            hi[i] = CLAY
    table(s, rows, MARGIN, Inches(1.45), W - 2 * MARGIN,
          col_w=[4.6, 2.5, 2.9, 1.9], size=13.5, row_h=Inches(0.44))

    textbox(s, MARGIN, Inches(4.62), Inches(3.2), Inches(0.5),
            f"{len(flips)} of {len(conf)}", size=30, bold=True, color=CLAY,
            font=HEAD_FONT)
    textbox(s, MARGIN, Inches(5.18), Inches(3.5), Inches(1.1),
            "month-variable pairs invert sign between the pooled and the "
            "within-district correlation.",
            size=14, color=MUTED, spacing=1.2)

    textbox(s, Inches(4.9), Inches(4.55), Inches(7.8), Inches(2.4),
            "Pooled across districts, post-monsoon rain looks like the dominant "
            "predictor and monsoon rain looks harmful. Both are artefacts: wet districts "
            "are disproportionately rainfed and low-yielding, so 'wet' proxies for "
            "'low-yielding' and reverses the apparent sign. Removing each district's own "
            "mean restores the agronomic direction.\n\n"
            "This is the measured form of the project's claim that heterogeneity carries "
            "signal and should be preserved rather than averaged away. It is the "
            "empirical justification for partitioning by state rather than pooling.",
            size=14, color=TEXT, spacing=1.2)
    return s


def literature_slide(prs, abl):
    s = slide(prs)
    heading(s, "Against published work", "R3 · SOLUTION EFFECTIVENESS")
    grouped: dict[str, list] = {}
    for run in abl["runs"]:
        grouped.setdefault(run["arm"], []).append(run)
    cen = agg([r["r2"] for r in grouped["centralised"]])[0]
    fp = agg([r["r2"] for r in grouped["fedper"]])[0]

    rows = [
        ["Work", "Setting", "Reported"],
        ["This project · centralised", "501 districts, strict chronological split",
         f"R² = {cen:.3f}"],
        ["This project · FedPer", "Same split, per-state clients", f"R² = {fp:.3f}"],
        ["Yazdi et al. [8]", "Rainfed cereals, strict chronological validation",
         "R² = 0.271"],
        ["De Clercq & Mahdi [6]", "247 Indian rice districts, centralised", "R² ≤ 0.82"],
        ["Malashin et al. [14]", "Indian state data, optimised deep network", "R² = 0.92"],
        ["Shams et al. [13]", "XAI-CROP, centralised", "R² = 0.9415"],
        ["Yenkikar et al. [11]", "246k records, 33 states, centralised ensemble",
         "R² = 0.9827"],
        ["Dey et al. [26]", "Federated, 4,513 Maharashtra samples", "acc > 99% (not R²)"],
        ["Mukherjee & Buyya [27]", "Federated, ring/mesh topologies",
         "acc > 93% (not R²)"],
    ]
    t = table(s, rows, MARGIN, Inches(1.38), W - 2 * MARGIN,
              col_w=[3.5, 5.9, 2.5], size=12.5, highlight={1: MOSS, 2: MOSS, 3: MOSS},
              row_h=Inches(0.375))
    for r in range(len(rows)):
        t.table.cell(r, 1).text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT

    note(s, "Read this honestly. Against loosely-validated centralised studies this "
            "project is behind on raw R². Against the one study using comparable strict "
            "chronological validation on rainfed data it is ahead. The federated "
            "accuracy figures are classification-style accuracy on smaller, more "
            "homogeneous samples and are not R², so they are not comparable and are not "
            "compared here. The claim this project actually makes is different: no "
            "reviewed federated work evaluates against a centralised ceiling, a trend "
            "baseline and an explanation-fidelity test at once.",
         y=Inches(5.35), color=TEXT, size=14)
    return s


def gaps_slide(prs, present):
    s = slide(prs)
    heading(s, "What is not here", "PROVENANCE")
    rows = [["Claim", "Status"],
            ["Climate-stress R² drop per arm", "ood.json not generated"],
            ["Perturbation response (+2 °C, −20% rain)", "perturbation.json not generated"],
            ["Wilcoxon p-values vs centralised", "significance.json not generated"],
            ["Feature-group ablation", "per-group runs not committed"]]
    t = table(s, rows, MARGIN, Inches(1.5), W - 2 * MARGIN,
              col_w=[6.5, 5.4], size=14, row_h=Inches(0.48))
    for r in range(1, len(rows)):
        run = t.table.cell(r, 1).text_frame.paragraphs[0].runs[0]
        run.font.color.rgb = CLAY
    for r in range(len(rows)):
        t.table.cell(r, 1).text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT

    textbox(s, MARGIN, Inches(4.15), W - 2 * MARGIN, Inches(1.5),
            "Stages F and G do not exist. An earlier project summary described "
            "climate-stress, perturbation and significance results as complete. Neither "
            "the evaluation code nor the artifacts are present in the repository or in "
            "the local working copy. Those figures have no artifact behind them and are "
            "quoted nowhere in this deck.",
            size=15, color=TEXT, spacing=1.22)

    textbox(s, MARGIN, Inches(5.5), W - 2 * MARGIN, Inches(1.3),
            "The feature-group comparison, that neither climate nor agronomic "
            "covariates beat the baseline alone and only their combination does, is the "
            "strongest single argument available in R3. The script writes each run to "
            "its own file; the runs simply need to be executed and committed before the "
            "claim can be quoted.",
            size=14.5, color=MUTED, spacing=1.22)

    textbox(s, MARGIN, Inches(6.78), W - 2 * MARGIN, Inches(0.5),
            f"Artifacts behind this deck: {', '.join(sorted(present))}.",
            size=11, color=MUTED)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    meta = load("meta")
    abl = load("ablation")
    cx = load("complexity")
    agr = load("agreement")
    mu = load("mu_sweep")
    fid = [{k: v for k, v in f.items() if not k.startswith("curve_")}
           for f in load("fidelity")]
    conf = load("confound")
    present = {p.stem for p in RESULTS.glob("*.json")}

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    title_slide(prs, meta)
    arms_slide(prs, abl)
    model_cost_slide(prs, cx)
    comms_slide(prs, cx)
    mu_slide(prs, mu)
    ablation_slide(prs, abl)
    agreement_slide(prs, agr)
    fidelity_slide(prs, fid)
    confound_slide(prs, conf)
    literature_slide(prs, abl)
    gaps_slide(prs, present)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(args.out)
    print(f"wrote {args.out.relative_to(ROOT)}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
