"""Figures 1--12 and Tables 1--6 for the current Best-of-N paper.

Single source of truth for the paper's visual identity. Every figure/table in
`plotting/figures.ipynb` is produced here so the notebook stays a
thin, readable narrative and the plotting code remains testable standalone.

Conventions:
  * Paul-Tol colourblind-safe palette, fixed per estimator across every figure.
  * log2 budget axis, nominal-level reference dotted, minimal spines, light grid.
  * Figures sized for a single-column page (~5.5in text width).
  * Tables are HTML with the best entry per metric highlighted in bold.
"""
from __future__ import annotations

import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
def _find_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "bon").is_dir():
            return p
    raise RuntimeError("repository root not found")


ROOT = _find_root()
SYN = ROOT / "plotting/data/synthetic"
REUSED = ROOT / "plotting/data/real"
INDEPENDENT = ROOT / "plotting/data/independent_nuisance"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
TABLE_OUT = ROOT / "tables"
TABLE_OUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Visual identity
# --------------------------------------------------------------------------
COL = {"dm": "#CC6677", "ipw": "#4477AA", "dr": "#228833", "reward_max": "#AA3377"}
LAB = {"dm": "Plug-in", "ipw": "BoN-IPW", "dr": "BoN-DR", "reward_max": "Naive BoN"}
ORACLE = "#000000"
EST3 = ["dm", "ipw", "dr"]
NOMINAL = 0.95

MODELS = ["gemma", "llama", "qwen"]
SELECTORS = ["oasst", "skywork"]
PRETTY_MODEL = {
    "gemma": "Gemma-2-2B-IT",
    "llama": "Llama-3.2-3B-Instruct",
    "qwen": "Qwen3.5-4B",
}
PRETTY_SEL = {"oasst": "OASST", "skywork": "Skywork"}
RM_NAME = {"oasst": "OASST-RM-2.1", "skywork": "Skywork-Reward-V2"}
RM_FULL = {"oasst": "OASST-RM-2.1-Pythia-1.4B",
           "skywork": "Skywork-Reward-V2-Qwen3-1.7B"}


def set_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.15, "grid.linewidth": 0.5,
        "lines.linewidth": 1.5, "lines.markersize": 3.5,
        "legend.frameon": False,
    })


def _budget_axis(ax, label="Best-of-N budget N"):
    ax.set_xscale("log", base=2)
    ax.set_xlabel(label)


def _row_label(ax, text):
    """Reward-model row label, placed left of the y-axis label."""
    ax.annotate(text, xy=(-0.40, 0.5), xycoords="axes fraction",
                rotation=90, ha="center", va="center", fontsize=8.5)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / (name + "." + ext), bbox_inches="tight")
    return OUT / (name + ".pdf")


# --------------------------------------------------------------------------
# Data loaders (cached)
# --------------------------------------------------------------------------
_cache = {}


def _load(key, fn):
    if key not in _cache:
        _cache[key] = fn()
    return _cache[key]


def syn_cases():
    return _load("sc", lambda: pd.read_csv(SYN / "cases.csv.gz"))


def syn_oracle():
    return _load("so", lambda: pd.read_csv(SYN / "oracles.csv.gz").merge(syn_cases(), on="case_id"))


def syn_calib():
    return _load("scal", lambda: pd.read_csv(SYN / "calibration_summary.csv.gz")
                 .merge(syn_cases(), on="case_id", suffixes=("", "_case")))


def syn_select():
    return _load("ssel", lambda: pd.read_csv(SYN / "selection_summary.csv.gz")
                 .merge(syn_cases(), on="case_id", suffixes=("", "_case")))


def syn_eff():
    return _load("seff", lambda: pd.read_csv(SYN / "efficiency_comparison.csv.gz")
                 .merge(syn_cases()[["case_id", "M"]], on="case_id", suffixes=("", "_case")))


def real_oracle():
    def f():
        o = pd.read_csv(REUSED / "oracles.csv.gz")
        return o[o.split == "test"].dropna(subset=["value"])
    return _load("ro", f)


def real_calib():
    def f():
        c = pd.read_csv(REUSED / "calibration_summary.csv.gz")
        return c[c.design == "iid_empirical"]
    return _load("rc", f)


def real_select():
    return _load("rs", lambda: pd.read_csv(REUSED / "selection_summary_prefixes.csv.gz"))


def real_overlap():
    return _load("rov", lambda: pd.read_csv(REUSED / "overlap.csv.gz"))


def independent_calib():
    return _load("ic", lambda: pd.read_csv(INDEPENDENT / "calibration_summary.csv"))


# --------------------------------------------------------------------------
# HTML tables with best-entry highlighting
# --------------------------------------------------------------------------
_TABLE_CSS = """
<style>
.pv3 {border-collapse:collapse; font-family:DejaVu Sans,Arial,sans-serif;
      font-size:12px; margin:6px 0 14px 0;}
.pv3 caption {caption-side:top; text-align:left; font-size:12px; padding:2px 0 6px 0;}
.pv3 th {border-bottom:1px solid #444; border-top:1px solid #444;
         padding:4px 9px; text-align:right; font-weight:600;}
.pv3 th.l, .pv3 td.l {text-align:left;}
.pv3 td {padding:3px 9px; text-align:right; border:none;}
.pv3 tr.last td {border-bottom:1px solid #444;}
.pv3 .best {font-weight:700;}
.pv3 tr.grp td {border-top:1px solid #ccc;}
.pv3note {font-size:11px; color:#444; padding:0 0 10px 0;
          font-family:DejaVu Sans,Arial,sans-serif; max-width:760px;}
</style>
"""
_last_table_frame: pd.DataFrame | None = None


def html_table(df, caption, group_cols=(), metric_specs=None, note="", float_fmt="{:.3f}"):
    """Render df as a paper-style HTML table.

    group_cols   : columns identifying a comparison group; 'best' is chosen
                   within each group across that group's rows.
    metric_specs : {column: 'min' | 'max' | ('close', target)}.
    """
    global _last_table_frame
    metric_specs = metric_specs or {}
    df = df.copy().reset_index(drop=True)
    _last_table_frame = df.copy()
    group_cols = list(group_cols)
    if group_cols:
        key = df[group_cols].astype(str).agg(" | ".join, axis=1)
    else:
        key = pd.Series([""] * len(df))

    best = {c: pd.Series([False] * len(df)) for c in metric_specs}
    for _, idx in df.groupby(key).groups.items():
        for c, spec in metric_specs.items():
            if c not in df.columns:
                continue
            vals = pd.to_numeric(df.loc[idx, c], errors="coerce")
            if vals.notna().sum() == 0:
                continue
            if spec == "min":
                tgt = vals.min()
            elif spec == "max":
                tgt = vals.max()
            else:
                order = (vals - spec[1]).abs().sort_values()
                tgt = vals.loc[order.index[0]]
            best[c].loc[idx] = vals.eq(tgt).fillna(False)

    cols = list(df.columns)
    # pandas >=3 reports string columns as 'str', not 'object'; test for
    # numeric instead so text cells are never sent through the float format.
    is_text = {c: not pd.api.types.is_numeric_dtype(df[c]) for c in cols}
    head = "".join(
        '<th class="l">' + str(c) + "</th>" if (c in group_cols or is_text[c])
        else "<th>" + str(c) + "</th>" for c in cols)

    rows, prev = [], None
    for i in range(len(df)):
        gk = key.iloc[i]
        cls = "grp" if (prev is not None and gk != prev) else ""
        prev = gk
        tds = []
        for c in cols:
            v = df.iloc[i][c]
            if is_text[c]:
                tds.append('<td class="l">' + str(v) + "</td>")
            else:
                txt = "—" if pd.isna(v) else float_fmt.format(v)
                mark = "best" if (c in best and bool(best[c].iloc[i])) else ""
                tds.append('<td class="' + mark + '">' + txt + "</td>")
        rows.append('<tr class="' + cls + '">' + "".join(tds) + "</tr>")
    if rows:
        rows[-1] = rows[-1].replace('<tr class="', '<tr class="last ', 1)

    note_html = '<div class="pv3note">' + note + "</div>" if note else ""
    return (_TABLE_CSS + '<table class="pv3"><caption>' + caption + "</caption>"
            + "<thead><tr>" + head + "</tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>" + note_html)


def save_table(rendered: str, name: str) -> str:
    """Save the most recently rendered paper table as CSV and styled HTML."""
    if _last_table_frame is None:
        raise RuntimeError("save_table must immediately follow a table renderer")
    if not name.startswith("table_") or not name.removeprefix("table_").isdigit():
        raise ValueError("table name must have the form table_NN")
    _last_table_frame.to_csv(TABLE_OUT / f"{name}.csv", index=False)
    (TABLE_OUT / f"{name}.html").write_text(rendered, encoding="utf-8")
    return rendered


def combo_label(model, selector):
    return PRETTY_MODEL[model] + " / " + PRETTY_SEL[selector]


# ==========================================================================
# Figure 1 (Introduction) - real-world motivation
# ==========================================================================
def fig1_real_regimes(N_max=256):
    """Show the OASST score/accuracy mismatch for the three reference models.

    Paper placement: Section 1 (motivation), cross-referenced from Section 5.3.
    No in-plot caption: row/column titles are data identifiers only.
    """
    o = real_oracle()
    o = o[o.N <= N_max]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.8), sharex=True)
    selector = "oasst"
    for j, model in enumerate(MODELS):
        ax = axes[j]
        g = o[(o.model == model) & (o.selector == selector)].sort_values("N")
        ax.axhline(g[g.N == 1].value.values[0], ls=":", color="0.6", lw=1)
        ax.plot(g.N, g.value, "o-", color=ORACLE, zorder=4)
        peak = g.loc[g.value.idxmax()]
        ax.scatter([peak.N], [peak.value], s=55, facecolors="none",
                   edgecolors="#CC3311", linewidths=1.5, zorder=6)
        near_right = peak.N >= 64
        ax.annotate("$N^{\\star}$=%d" % int(peak.N), (peak.N, peak.value),
                    textcoords="offset points", xytext=(-7 if near_right else 7, -10),
                    ha="right" if near_right else "left", va="top",
                    color="#CC3311", fontsize=7.5,
                    bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.85))
        ax.margins(y=0.20)
        score_axis = ax.twinx()
        score_axis.plot(g.N, g.expected_score, "--", color=COL["reward_max"], lw=1.2)
        score_axis.tick_params(axis="y", colors=COL["reward_max"], labelsize=6.5)
        score_axis.grid(False)
        score_axis.spines["right"].set_visible(True)
        score_axis.spines["right"].set_color(COL["reward_max"])
        if j == 2:
            score_axis.set_ylabel("selected score", color=COL["reward_max"], fontsize=7.5)
        ax.set_xscale("log", base=2)
        ax.set_title(PRETTY_MODEL[model])
        ax.set_xlabel("$N$")
        if j == 0:
            ax.set_ylabel("Task accuracy\nOASST-RM-2.1")
    fig.tight_layout()
    save(fig, "figure_01")
    return fig


# ==========================================================================
# RQ1 - synthetic study
# ==========================================================================
def fig2_syn_regimes(gammas=(0.0, 0.60, 0.65, 0.80), delta=0.4, n_val=4096):
    """Synthetic DGP spans aligned -> interior-optimum -> severe regimes."""
    o = syn_oracle()
    o = o[(o.nuisance_distortion == delta) & (o.n_val == n_val) & (o.M == 256)]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.0))
    cmap = plt.get_cmap("viridis")
    cols = {g: cmap(i / max(1, len(gammas) - 1) * 0.85) for i, g in enumerate(gammas)}
    for g in gammas:
        q = o[o.score_distortion == g].sort_values("N")
        if q.empty:
            continue
        axes[0].plot(q.N, q.truth, "o-", color=cols[g], label="%g" % g)
        pk = q.loc[q.truth.idxmax()]
        axes[0].scatter([pk.N], [pk.truth], s=50, facecolors="none",
                        edgecolors=cols[g], linewidths=1.4, zorder=5)
        axes[1].plot(q.N, q.expected_score, "o-", color=cols[g])
    axes[0].set_ylabel("oracle policy value")
    axes[1].set_ylabel("expected selected score")
    for ax in axes:
        _budget_axis(ax, "$N$")
    axes[0].legend(title="score error $\\gamma$", ncol=2, loc="lower left", fontsize=6.5,
                   title_fontsize=6.5)
    fig.tight_layout()
    save(fig, "figure_05")
    return fig


def fig3_double_robustness(gamma=0.65, n_val=4096, N_show=256, rule="improvement_lcb",
                           N_max=256, deltas=(0.0, 0.2, 0.4, 0.6)):
    """Double robustness: BoN-DR/BoN-IPW are invariant to reward function error.

    Evaluated at the primary endpoint N=256 (not a hand-picked budget); the
    pattern is qualitatively identical at every budget, see Table 1.
    """
    cal = syn_calib()
    sel = syn_select()
    c = cal[(cal.sweeps.str.contains("nuisance")) & (cal.N == N_show)
            & (cal.score_distortion == gamma) & (cal.n_val == n_val)
            & (cal.nuisance_distortion.isin(deltas))]
    s = sel[(sel.sweeps.str.contains("nuisance")) & (sel.N_max == N_max)
            & (sel.selection_rule == rule) & (sel.score_distortion == gamma)
            & (sel.n_val == n_val) & (sel.nuisance_distortion.isin(deltas))]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.85))
    for e in EST3:
        ce = c[c.estimator == e].sort_values("nuisance_distortion")
        axes[0].plot(ce.nuisance_distortion, ce.bias.abs(), "o-", color=COL[e], label=LAB[e])
        axes[1].plot(ce.nuisance_distortion, ce.pointwise_coverage, "o-", color=COL[e])
        se = s[s.estimator == e].sort_values("nuisance_distortion")
        axes[2].plot(se.nuisance_distortion, se.mean_gain, "o-", color=COL[e])
    axes[1].axhline(NOMINAL, ls=":", color="0.35", lw=1)
    axes[2].axhline(0, ls=":", color="0.35", lw=1)
    axes[0].set_ylabel("|bias|")
    axes[1].set_ylabel("95% CI coverage")
    axes[1].set_ylim(-0.03, 1.03)
    axes[2].set_ylabel("value gain over $N$=1")
    for ax in axes:
        ax.set_xlabel("reward function error $\\delta$")
        ax.set_xticks(list(deltas))
    axes[0].legend(loc="upper left", fontsize=6.5)
    fig.tight_layout()
    save(fig, "figure_02")
    return fig


def fig4_consistency(gamma=0.65, delta=0.4, N_show=256, rule="value", N_max=256):
    """Consistency in n: BoN-DR/BoN-IPW converge, Plug-in does not."""
    cal = syn_calib()
    sel = syn_select()
    c = cal[(cal.sweeps.str.contains("n_val")) & (cal.N == N_show)
            & (cal.score_distortion == gamma) & (cal.nuisance_distortion == delta)]
    s = sel[(sel.sweeps.str.contains("n_val")) & (sel.N_max == N_max)
            & (sel.selection_rule == rule) & (sel.score_distortion == gamma)
            & (sel.nuisance_distortion == delta)]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.85))
    for e in EST3:
        ce = c[c.estimator == e].sort_values("n_val")
        axes[0].plot(ce.n_val, ce.rmse, "o-", color=COL[e], label=LAB[e])
        axes[1].plot(ce.n_val, ce.pointwise_coverage, "o-", color=COL[e])
        se = s[s.estimator == e].sort_values("n_val")
        axes[2].plot(se.n_val, se.oracle_optimal_rate, "o-", color=COL[e])
    axes[1].axhline(NOMINAL, ls=":", color="0.35", lw=1)
    axes[0].set_yscale("log")
    # plain decimal RMSE ticks instead of matplotlib's 4x10^-1 notation
    _rmse_ticks = [0.03, 0.05, 0.10, 0.20, 0.40]
    axes[0].yaxis.set_major_locator(mpl.ticker.FixedLocator(_rmse_ticks))
    axes[0].yaxis.set_major_formatter(
        mpl.ticker.FixedFormatter(["%.2f" % v for v in _rmse_ticks]))
    axes[0].yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    axes[0].set_ylabel("RMSE")
    axes[1].set_ylabel("95% CI coverage")
    axes[1].set_ylim(-0.03, 1.03)
    axes[2].set_ylabel("P[select oracle-best $N$]")
    axes[2].set_ylim(-0.03, 1.03)
    for ax in axes:
        ax.set_xscale("log", base=10)
        ax.set_xlabel("validation size $n$")
    axes[0].legend(loc="lower left", fontsize=6.5)
    fig.tight_layout()
    save(fig, "figure_06")
    return fig


# ==========================================================================
# RQ2 - real-world off-policy evaluation
# ==========================================================================
Z80 = 1.2816


def fig5_ope_overlay(N_max=256):
    """Estimated vs. true BoN value curves on GSM8K, with 80% intervals.

    This is the central RQ2 exhibit: Plug-in drifts away from the oracle even
    where the reward model is well aligned, while BoN-DR and BoN-IPW track it -- BoN-DR with
    visibly tighter intervals.
    """
    c = real_calib()
    c = c[c.N <= N_max]
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.2), sharex=True)
    for i, sel in enumerate(SELECTORS):
        for j, mod in enumerate(MODELS):
            ax = axes[i, j]
            sub = c[(c.model == mod) & (c.selector == sel)]
            truth = sub.groupby("N").truth.mean().sort_index()
            ax.plot(truth.index, truth.values, "--", color=ORACLE, lw=1.8,
                    label="Oracle", zorder=6)
            lo_r, hi_r = [truth.min()], [truth.max()]
            for e in ["ipw", "dr", "dm"]:
                q = sub[sub.estimator == e].sort_values("N")
                ax.plot(q.N, q.mean_estimate, "o-", color=COL[e], ms=2.8, lw=1.3,
                        label=LAB[e], zorder=5 if e == "dm" else 4)
                lo_r.append(q.mean_estimate.min())
                hi_r.append(q.mean_estimate.max())
                if e in ("dr", "ipw"):
                    ax.fill_between(q.N, q.mean_estimate - Z80 * q.mean_se,
                                    q.mean_estimate + Z80 * q.mean_se,
                                    color=COL[e], alpha=0.16, lw=0, zorder=2)
            lo, hi = min(lo_r), max(hi_r)
            pad = 0.22 * (hi - lo + 1e-9)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xscale("log", base=2)
            if i == 0:
                ax.set_title(PRETTY_MODEL[mod])
            if j == 0:
                ax.set_ylabel(RM_NAME[sel], fontsize=8.5)
            if i == 1:
                ax.set_xlabel("$N$")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=7.5,
               bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.tight_layout()
    fig.supylabel("Task accuracy", fontsize=8.5, x=-0.01)
    save(fig, "figure_03")
    return fig


def fig6_real_coverage(N_max=256):
    """Pointwise interval validity across budgets on GSM8K."""
    c = real_calib()
    c = c[c.N <= N_max]
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.0), sharex=True, sharey=True)
    for i, sel in enumerate(SELECTORS):
        for j, mod in enumerate(MODELS):
            ax = axes[i, j]
            for e in EST3:
                q = c[(c.model == mod) & (c.selector == sel)
                      & (c.estimator == e)].sort_values("N")
                ax.plot(q.N, q.pointwise_coverage, "o-", color=COL[e], label=LAB[e])
            ax.axhline(NOMINAL, ls=":", color="0.35", lw=1)
            ax.set_xscale("log", base=2)
            ax.set_ylim(-0.04, 1.04)
            if i == 0:
                ax.set_title(PRETTY_MODEL[mod])
            if j == 0:
                ax.set_ylabel(RM_NAME[sel], fontsize=8.5)
            if i == 1:
                ax.set_xlabel("$N$")
    axes[0, 0].legend(loc="lower left", fontsize=6.5)
    fig.tight_layout()
    fig.supylabel("95% CI coverage", fontsize=8.5, x=-0.01)
    save(fig, "figure_08")
    return fig


# ==========================================================================
# RQ3 - uncertainty-aware selection (synthetic + real)
# ==========================================================================
_RULE_LAB = {"value": "Value-max", "improvement_lcb": "Improvement-LCB",
             "reward_max": "Naive BoN"}


_EST_ORDER = {"dm": 0, "ipw": 1, "dr": 2, "reward_max": 3}


def _order_est(df):
    df = df.copy()
    df["_o"] = df.estimator.map(_EST_ORDER)
    return df.sort_values("_o").drop(columns="_o")


def table1_controlled(gamma=0.65, delta=0.4, n_val=4096, Ns=(8, 256)):
    """Synthetic experiments calibration at the moderate regime."""
    c = syn_calib()
    c = c[(c.score_distortion == gamma) & (c.nuisance_distortion == delta)
          & (c.n_val == n_val) & (c.M == 256) & (c.N.isin(Ns))]
    rows = []
    for N in Ns:
        for _, r in _order_est(c[c.N == N]).iterrows():
            rows.append({"N": int(N), "estimator": LAB[r.estimator],
                         "bias": r.bias, "RMSE": r.rmse,
                         "mean SE": r.mean_se, "CI width": r.mean_ci_width,
                         "coverage": r.pointwise_coverage})
    df = pd.DataFrame(rows)
    return html_table(
        df, "Table 1. Synthetic experiments calibration under joint score and reward function "
            "error (score error %.2f, reward function error %.1f, n=%s, 300 "
            "repetitions)." % (gamma, delta, format(n_val, ",")),
        group_cols=["N"],
        metric_specs={"RMSE": "min", "coverage": ("close", NOMINAL)},
        note="Bold = best per budget. Plug-in is the most precise estimator "
             "(narrowest interval) yet never covers, because its interval is "
             "centred on a biased target; its bias grows from 0.049 at N=8 to "
             "0.369 at N=256. BoN-DR attains the lowest RMSE among valid estimators.")


# ==========================================================================
# Appendix
# ==========================================================================
def figA1_bias_variance(N_max=256):
    """Bias / dispersion / RMSE over the budget grid on GSM8K (Appendix).

    Shows the trade-off that a single-budget table hides: BoN-DR and BoN-IPW are
    essentially unbiased at every budget, but their dispersion grows as overlap
    deteriorates, so Plug-in's flat bias is only competitive in RMSE at the
    largest budgets.
    """
    c = real_calib()
    c = c[c.N <= N_max]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.9))
    for e in EST3:
        g = c[c.estimator == e].groupby("N")
        axes[0].plot(g.bias.mean().index, g.bias.mean().abs().values, "o-",
                     color=COL[e], label=LAB[e])
        axes[1].plot(g.empirical_sd.mean().index, g.empirical_sd.mean().values,
                     "o-", color=COL[e])
        axes[2].plot(g.rmse.mean().index, g.rmse.mean().values, "o-", color=COL[e])
    axes[0].set_ylabel("|bias|")
    axes[1].set_ylabel("empirical SD")
    axes[2].set_ylabel("RMSE")
    for ax in axes:
        _budget_axis(ax, "$N$")
        ax.set_yscale("log")
    axes[0].legend(loc="lower right", fontsize=6.5)
    fig.tight_layout()
    save(fig, "figure_09")
    return fig


def figA2_efficiency(gamma=0.65, n_val=4096):
    """Finite-M efficiency: BoN-DR approaches the semiparametric bound as M grows."""
    eff = syn_eff()
    cal = syn_calib()
    c = cal[(cal.sweeps.str.contains("M")) & (cal.score_distortion == gamma)
            & (cal.nuisance_distortion == 0.0) & (cal.n_val == n_val)]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.0))
    Ms = sorted(eff.M.unique())
    cmap = plt.get_cmap("viridis")
    for i, M in enumerate(Ms):
        q = eff[eff.M == M].sort_values("N")
        axes[0].plot(q.N, q.paired_variance_ratio, "o-",
                     color=cmap(i / max(1, len(Ms) - 1) * 0.85), label="%d" % M)
    axes[0].axhline(1.0, ls=":", color="0.35", lw=1)
    axes[0].set_ylabel("finite-$M$ / infinite-$M$ variance")
    axes[0].legend(title="$M$", ncol=2, fontsize=6.5, title_fontsize=6.5)
    for e in ["dr", "ipw"]:
        q = c[(c.estimator == e) & (c.M == 256)].sort_values("N")
        axes[1].plot(q.N, q.variance_over_efficiency_bound, "o-", color=COL[e],
                     label=LAB[e])
    axes[1].axhline(1.0, ls=":", color="0.35", lw=1)
    axes[1].set_ylabel("variance / efficiency bound")
    axes[1].legend(fontsize=6.5)
    for ax in axes:
        _budget_axis(ax, "$N$")
    fig.tight_layout()
    save(fig, "figure_07")
    return fig


def figA3_overlap():
    """Effective sample size collapse explains the large-budget regime."""
    o = real_overlap()
    t = o.groupby(["model", "selector", "N"]).agg(
        ess=("ess", "mean"), nz=("nonzero_rate", "mean")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.0))
    for sl, ls in zip(SELECTORS, ["-", "--"]):
        for m in MODELS:
            q = t[(t.model == m) & (t.selector == sl)].sort_values("N")
            axes[0].plot(q.N, q.ess, ls, color="0.25" if sl == "oasst" else "0.6",
                         label=PRETTY_SEL[sl] if m == "gemma" else None)
            axes[1].plot(q.N, q.nz, ls, color="0.25" if sl == "oasst" else "0.6")
    for ax in axes:
        ax.axvline(256, color="#CC3311", lw=1, ls=":")
        _budget_axis(ax, "$N$")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("effective sample size")
    axes[1].set_ylabel("non-zero weight rate")
    axes[0].legend(title="reward model", fontsize=6.5, title_fontsize=6.5)
    fig.tight_layout()
    save(fig, "figure_10")
    return fig


def figA4_syn_score_selection(N_max=256, rule="improvement_lcb"):
    """Synthetic experiments selection across overoptimisation strength."""
    s = syn_select()
    s = s[(s.sweeps.str.contains("score")) & (s.N_max == N_max)
          & (s.nuisance_distortion == 0.4) & (s.n_val == 4096)]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.0))
    rm = s[s.selection_rule == "reward_max"].sort_values("score_distortion")
    axes[0].plot(rm.score_distortion, rm.mean_gain, "s--", color=COL["reward_max"],
                 label=LAB["reward_max"])
    axes[1].plot(rm.score_distortion, rm.mean_oracle_regret, "s--", color=COL["reward_max"])
    for e in EST3:
        q = s[(s.selection_rule == rule) & (s.estimator == e)].sort_values("score_distortion")
        axes[0].plot(q.score_distortion, q.mean_gain, "o-", color=COL[e], label=LAB[e])
        axes[1].plot(q.score_distortion, q.mean_oracle_regret, "o-", color=COL[e])
    axes[0].axhline(0, ls=":", color="0.35", lw=1)
    axes[0].set_ylabel("value gain over $N$=1")
    axes[1].set_ylabel("oracle regret")
    for ax in axes:
        ax.set_xlabel("score error $\\gamma$")
    axes[0].legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    save(fig, "figure_12")
    return fig


def tableA1_real_estimation(N_show=256):
    """Per-configuration estimation summary at the primary endpoint (Appendix)."""
    c = real_calib()
    c = c[c.N == N_show]
    rows = []
    for sl in SELECTORS:
        for m in MODELS:
            for _, r in _order_est(c[(c.model == m) & (c.selector == sl)]).iterrows():
                rows.append({"reward model": PRETTY_SEL[sl],
                             "behaviour model": PRETTY_MODEL[m],
                             "estimator": LAB[r.estimator],
                             "bias": r.bias, "RMSE": r.rmse,
                             "CI width": r.mean_ci_width,
                             "coverage": r.pointwise_coverage})
    df = pd.DataFrame(rows)
    return html_table(
        df, "Table A1. GSM8K estimation at N=%d (500 repetitions, n=3,973). This is "
            "the most demanding budget in the primary grid; see Figure A1 for the "
            "full budget range." % N_show,
        group_cols=["reward model", "behaviour model"],
        metric_specs={"RMSE": "min", "coverage": ("close", NOMINAL)},
        note="Bold = best per configuration. Plug-in coverage is exactly zero "
             "everywhere. BoN-DR gives the lowest RMSE in every configuration; BoN-IPW is "
             "best calibrated at this endpoint.")


def tableA2_independent_nuisance(N_show=256):
    """Independently trained reward functions (Appendix)."""
    c = independent_calib()
    c = c[(c.selector == "oasst") & (c.N == N_show)]
    rows = []
    for m in MODELS:
        for nu in sorted(c.nuisance.unique()):
            for _, r in _order_est(c[(c.model == m) & (c.nuisance == nu)]).iterrows():
                rows.append({"behaviour model": PRETTY_MODEL[m],
                             "reward function": nu.replace("_", " "),
                             "estimator": LAB[r.estimator],
                             "bias": r.bias, "RMSE": r.rmse,
                             "coverage": r.pointwise_coverage})
    df = pd.DataFrame(rows)
    return html_table(
        df, "Table A2. Robustness to the reward function: independently trained "
            "TF-IDF and embedding-LightGBM nuisances (OASST, N=%d)." % N_show,
        group_cols=["behaviour model", "reward function"],
        metric_specs={"RMSE": "min", "coverage": ("close", NOMINAL)},
        note="Plug-in fails to cover for every reward function, so its invalidity "
             "is not an artefact of the score-calibration spline used in the main text.")


# ==========================================================================
# RQ3 (Section 5.4) - selection under *challenging* conditions
#
# Main text deliberately shows only the two hard regimes and only the two rules
# the paper actually proposes (value-max and improvement-LCB). The plug-in is
# shown with value-max only: an LCB built on the plug-in is not part of the
# method and would be an artificial construct. Aligned/easy configurations
# live in the appendix.
# ==========================================================================

# (estimator, rule, display label) in fixed presentation order
_S54_METHODS = [
    ("reward_max", "reward_max", "Naive BoN"),
    ("dm", "value", "Plug-in"),
    ("ipw", "value", "BoN-IPW, Value-max"),
    ("ipw", "improvement_lcb", "BoN-IPW, Improvement-LCB"),
    ("dr", "value", "BoN-DR, Value-max"),
    ("dr", "improvement_lcb", "BoN-DR, Improvement-LCB"),
]

# Synthetic experiments challenging setting: overoptimised score AND misspecified nuisance
S54_GAMMA, S54_DELTA, S54_N = 0.65, 0.6, 4096
S54_MODEL, S54_SELECTOR = "qwen", "oasst"


def _s54_syn(N_max=256):
    s = syn_select()
    return s[(s.N_max == N_max) & (s.score_distortion == S54_GAMMA)
             & (s.nuisance_distortion == S54_DELTA) & (s.n_val == S54_N)]


def _s54_real(N_max=256):
    s = real_select()
    return s[(s.N_max == N_max) & (s.model == S54_MODEL)
             & (s.selector == S54_SELECTOR)]


def _s54_oracle_gain_syn():
    """Best achievable gain over N=1 in the Synthetic experiments challenging setting."""
    o = syn_oracle()
    g = o[(o.score_distortion == S54_GAMMA) & (o.nuisance_distortion == S54_DELTA)
          & (o.n_val == S54_N) & (o.M == 256)].sort_values("N")
    if g.empty:  # delta=0.6 case may carry a different pool size
        g = o[(o.score_distortion == S54_GAMMA)
              & (o.nuisance_distortion == S54_DELTA)].sort_values("N")
    return g.truth.max() - g[g.N == 1].truth.values[0]


def _s54_oracle_gain_real(N_max=256):
    o = real_oracle()
    g = o[(o.model == S54_MODEL) & (o.selector == S54_SELECTOR)
          & (o.N <= N_max)].sort_values("N")
    return 100 * (g.value.max() - g[g.N == 1].value.values[0])


def _s54_rows(frame, scale, setting):
    rows = []
    for est, rule, lab in _S54_METHODS:
        q = frame[(frame.estimator == est) & (frame.selection_rule == rule)]
        if not len(q):
            continue
        r = q.iloc[0]
        rows.append({"setting": setting, "method": lab,
                     "gain": scale * r.mean_gain, "harm rate": r.harm_rate,
                     "median N": r.median_selected_N})
    return rows


def table2_selection_challenging(N_max=256):
    """Main-text Section 5.4 table: both challenging regimes, one view."""
    rows = _s54_rows(_s54_syn(N_max), 1.0, "Synthetic experiments")
    rows.append({"setting": "Synthetic experiments", "method": "oracle best",
                 "gain": _s54_oracle_gain_syn(), "harm rate": 0.0, "median N": np.nan})
    rows += _s54_rows(_s54_real(N_max), 100.0, "GSM8K Qwen / OASST")
    rows.append({"setting": "GSM8K Qwen / OASST", "method": "oracle best",
                 "gain": _s54_oracle_gain_real(N_max), "harm rate": 0.0,
                 "median N": np.nan})
    df = pd.DataFrame(rows)
    return html_table(
        df, "Table 2. Selecting N under challenging conditions. Synthetic experiments: "
            "score error %.2f and reward function error %.1f, n=%s, 300 repetitions "
            "(gain in policy value). GSM8K: Qwen3.5-4B with the exploitable "
            "OASST-RM-2.1, n=3,973, 100 logging repetitions (gain in accuracy "
            "points). N_max=%d."
            % (S54_GAMMA, S54_DELTA, format(S54_N, ","), N_max),
        group_cols=["setting"],
        metric_specs={"gain": "max", "harm rate": "min"},
        note="Bold = best per setting, excluding the oracle reference. In the "
             "Synthetic experiments setting the sample is large enough that Value-max already "
             "suffices: BoN-IPW and BoN-DR recover most of the achievable gain with no harm, "
             "and the LCB rule is only slightly more conservative. On GSM8K the "
             "sample is smaller and overlap is far worse, so Value-max alone is not "
             "enough -- it still harms 73-77% of runs -- while Improvement-LCB "
             "removes nearly all of it. Naive BoN and Plug-in harm every "
             "run in both settings.")


def fig7_selection_challenging(N_max=256):
    """Main-text Section 5.4 figure: gain and harm in the two hard regimes."""
    panels = [(_s54_syn(N_max), 1.0, "value gain", _s54_oracle_gain_syn(), "%.3f"),
              (_s54_real(N_max), 100.0, "accuracy gain (pp)",
               _s54_oracle_gain_real(N_max), "%.2f")]
    labels = [lab for _, _, lab in _S54_METHODS]
    colors = [COL[e] for e, _, _ in _S54_METHODS]
    alphas = [1.0 if r != "value" or e in ("reward_max", "dm") else 0.55
              for e, r, _ in _S54_METHODS]
    x = np.arange(len(_S54_METHODS))
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.4))
    for i, (frame, scale, ylab, oracle, gfmt) in enumerate(panels):
        gains, harms = [], []
        for est, rule, _ in _S54_METHODS:
            q = frame[(frame.estimator == est) & (frame.selection_rule == rule)]
            gains.append(scale * q.mean_gain.values[0] if len(q) else np.nan)
            harms.append(q.harm_rate.values[0] if len(q) else np.nan)
        for k, vals in enumerate([gains, harms]):
            ax = axes[i, k]
            fmt = gfmt if k == 0 else "%.2f"
            for b in range(len(x)):
                ax.bar(x[b], vals[b], 0.68, color=colors[b], alpha=alphas[b])
                v = vals[b]
                if np.isfinite(v):
                    ax.annotate(fmt % v, (x[b], v), textcoords="offset points",
                                xytext=(0, 2.5 if v >= 0 else -2.5), ha="center",
                                va="bottom" if v >= 0 else "top", fontsize=5.8,
                                color="0.25")
            ax.set_xticks(x)
            ax.set_xticklabels(labels if i == 1 else [], rotation=28,
                               ha="right", fontsize=7)
            ax.grid(axis="x", alpha=0)
            if k == 0:
                # Headroom so the reference lines are never flush against the
                # frame: on GSM8K the best attainable gain IS zero, so that line
                # carries the message and must be clearly readable.
                finite = [v for v in vals if np.isfinite(v)]
                lo = min([0.0, oracle] + finite)
                hi = max([0.0, oracle] + finite)
                rng = (hi - lo) or 1.0
                ax.set_ylim(lo - 0.16 * rng, hi + 0.34 * rng)
                if abs(oracle) < 1e-9:
                    # oracle-best coincides with the N=1 baseline: draw one line
                    ax.axhline(0.0, ls=(0, (4, 2)), color="#CC3311", lw=1.6, zorder=5)
                    note = "best attainable = $N$=1"
                else:
                    ax.axhline(0, color="0.45", lw=0.9)
                    ax.axhline(oracle, ls=(0, (4, 2)), color="#CC3311", lw=1.6, zorder=5)
                    note = "best attainable"
                ax.annotate(note, xy=(0.015, oracle),
                            xycoords=ax.get_yaxis_transform(),
                            ha="left", va="bottom", fontsize=6.5, color="#CC3311")
                ax.set_ylabel(ylab, fontsize=8)
            else:
                ax.set_ylim(0, 1.16)
                ax.set_ylabel("harm rate", fontsize=8)
                ax.annotate("Synthetic experiments" if i == 0 else "GSM8K",
                            xy=(1.05, 0.5), xycoords="axes fraction", rotation=270,
                            ha="center", va="center", fontsize=8)
    fig.tight_layout()
    save(fig, "figure_11")
    return fig


# --------------------------------------------------------------------------
# Current manuscript numbering
# --------------------------------------------------------------------------
def figure_4(N_max=256):
    """Figure 4: GSM8K harm rate for the paper's six selection methods."""
    methods = [
        ("reward_max", "reward_max", "Naive BoN"),
        ("dm", "value", "Plug-in"),
        ("ipw", "value", "BoN-IPW, Value-max"),
        ("ipw", "improvement_lcb", "BoN-IPW, Improvement-LCB"),
        ("dr", "value", "BoN-DR, Value-max"),
        ("dr", "improvement_lcb", "BoN-DR, Improvement-LCB"),
    ]
    frame = _s54_real(N_max)
    values = []
    for estimator, rule, _ in methods:
        row = frame[(frame.estimator == estimator) & (frame.selection_rule == rule)]
        values.append(row.harm_rate.iloc[0] if len(row) else np.nan)
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(3.7, 2.7))
    for index, ((estimator, rule, _), value) in enumerate(zip(methods, values, strict=True)):
        alpha = 0.55 if rule == "value" and estimator in {"ipw", "dr"} else 1.0
        ax.bar(index, value, 0.68, color=COL[estimator], alpha=alpha)
        ax.annotate(f"{value:.2f}", (index, value), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=6)
    ax.set_xticks(x, [label for _, _, label in methods], rotation=28, ha="right")
    ax.set_ylabel("harm rate")
    ax.set_ylim(0, 1.16)
    ax.grid(axis="x", alpha=0)
    fig.tight_layout()
    save(fig, "figure_04")
    return fig


def table_1():
    """Table 1: synthetic estimation and calibration."""
    return table1_controlled()


def table_2():
    """Table 2: GSM8K estimation at N=256."""
    return tableA1_real_estimation().replace("Table A1.", "Table 2.")


def table_3():
    """Table 3: GSM8K estimation with independently trained reward functions."""
    return tableA2_independent_nuisance().replace("Table A2.", "Table 3.")


def table_4():
    """Table 4: selection under challenging conditions."""
    return table2_selection_challenging().replace("Table 2.", "Table 4.")


def table_5(N_max=256):
    """Table 5: complete GSM8K selection results in manuscript form."""
    source = real_select()
    source = source[(source.N_max == N_max) & source.selection_rule.isin(
        ["reward_max", "value", "improvement_lcb"]
    )]
    rows = []
    for selector in SELECTORS:
        for model in MODELS:
            block = source[(source.selector == selector) & (source.model == model)]
            naive = block[block.selection_rule == "reward_max"]
            if len(naive):
                row = naive.iloc[0]
                rows.append({"reward model": RM_NAME[selector], "reference model": PRETTY_MODEL[model],
                             "rule": "Naive BoN", "estimator": "Naive BoN",
                             "gain (pp)": 100 * row.mean_gain, "harm": row.harm_rate})
            for rule in ("value", "improvement_lcb"):
                for estimator in EST3:
                    match = block[(block.selection_rule == rule) & (block.estimator == estimator)]
                    if len(match):
                        row = match.iloc[0]
                        rows.append({"reward model": RM_NAME[selector],
                                     "reference model": PRETTY_MODEL[model],
                                     "rule": _RULE_LAB[rule], "estimator": LAB[estimator],
                                     "gain (pp)": 100 * row.mean_gain, "harm": row.harm_rate})
    return html_table(pd.DataFrame(rows),
        "Table 5. Complete selection results on GSM8K.",
        group_cols=["reward model", "reference model"],
        metric_specs={"gain (pp)": "max", "harm": "min"})


def table_6(N_max=256, n_val=4096, gamma=0.65):
    """Table 6: complete synthetic selection results in manuscript form."""
    source = syn_select()
    source = source[(source.N_max == N_max) & (source.n_val == n_val)
                    & (source.score_distortion == gamma)
                    & source.selection_rule.isin(["reward_max", "value", "improvement_lcb"])]
    rows = []
    for delta in sorted(source.nuisance_distortion.unique()):
        block = source[source.nuisance_distortion == delta]
        naive = block[block.selection_rule == "reward_max"]
        if len(naive):
            row = naive.iloc[0]
            rows.append({"reward function error": delta, "rule": "Naive BoN",
                         "estimator": "Naive BoN", "gain": row.mean_gain,
                         "harm": row.harm_rate})
        for rule in ("value", "improvement_lcb"):
            for estimator in EST3:
                match = block[(block.selection_rule == rule) & (block.estimator == estimator)]
                if len(match):
                    row = match.iloc[0]
                    rows.append({"reward function error": delta, "rule": _RULE_LAB[rule],
                                 "estimator": LAB[estimator], "gain": row.mean_gain,
                                 "harm": row.harm_rate})
    return html_table(pd.DataFrame(rows),
        "Table 6. Complete selection results in the Synthetic experiments.",
        group_cols=["reward function error"], metric_specs={"gain": "max", "harm": "min"})


# Paper figure numbering.
figure_1 = fig1_real_regimes
figure_2 = fig3_double_robustness
figure_3 = fig5_ope_overlay
figure_5 = fig2_syn_regimes
figure_6 = fig4_consistency
figure_7 = figA2_efficiency
figure_8 = fig6_real_coverage
figure_9 = figA1_bias_variance
figure_10 = figA3_overlap
figure_11 = fig7_selection_challenging
figure_12 = figA4_syn_score_selection
