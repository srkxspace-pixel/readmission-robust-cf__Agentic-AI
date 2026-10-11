"""Paper figures from results/eval/*.csv only (SPECS FR-7). Static print figures: light surface, validated
categorical slots 1-3 for the three methods (dataviz reference palette), gray for reference marks.

Run: python -m rcf.reporting.figures  ->  figures/*.pdf + *.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from rcf.utils import ROOT, load_config  # noqa: E402

import os  # noqa: E402

_SMALL = os.environ.get("RCF_SMALL") == "1"  # smoke test: read results/eval_small, write there
EVAL = ROOT / "results" / ("eval_small" if _SMALL else "eval")
FIG = EVAL / "figures" if _SMALL else ROOT / "figures"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
TAU = load_config("counterfactuals")["consensus"]["tau"]
CONS = f"Consensus (tau={TAU})"
METHODS = ["DiCE", CONS, "ElliCE"]                       # fixed order -> fixed colour (never by rank)
COLOR = {"DiCE": "#2a78d6", CONS: "#eb6834", "ElliCE": "#1baf7a"}
LABEL = {"DiCE": "DiCE", CONS: f"Consensus (τ={TAU})", "ElliCE": "ElliCE"}
NICE = {"cross_model_validity": "Cross-model validity", "feature_set_consistency": "Feature-set consistency",
        "direction_consistency": "Direction consistency", "magnitude_consistency": "Magnitude consistency",
        "time_in_hospital": "Time in hospital", "num_lab_procedures": "Lab procedures",
        "num_procedures": "Procedures", "num_medications": "Medications"}

plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "legend.frameon": False, "axes.spines.top": False,
                     "axes.spines.right": False})


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_pipeline():
    steps = ["Diabetes 130-US\n69,970 patients", "Patient-level\n80/20 split", "15 models\n3 families × 5",
             "Cohorts\nstability,\ncoverage", "CFs: DiCE,\nConsensus,\nElliCE", "Cross-evaluate\nin-sample\n+ LOFO",
             "4 stability +\nquality metrics,\nbootstrap CIs"]
    fig, ax = plt.subplots(figsize=(7.16, 1.15))
    ax.set_axis_off()
    w, gap = 0.124, 0.022
    for i, s in enumerate(steps):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.1), w, 0.8, boxstyle="round,pad=0.005,rounding_size=0.02",
                                    fc="#f0efec", ec=MUTED, lw=0.6))
        ax.text(x + w / 2, 0.5, s, ha="center", va="center", fontsize=5.8, color=INK, linespacing=1.15)
        if i < len(steps) - 1:
            ax.annotate("", (x + w + gap, 0.5), (x + w, 0.5), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.7))
    ax.set_xlim(-0.01, len(steps) * (w + gap) - gap + 0.01)
    ax.set_ylim(0, 1)
    save(fig, "fig1_pipeline")


def fig_stability(methods):
    """Small multiples: one panel per thesis metric; methods on y; filled = in-sample, open = LOFO."""
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.7), sharey=True, gridspec_kw={"wspace": 0.18})
    for ax, metric in zip(axes, NICE):
        if not metric.endswith("consistency") and metric != "cross_model_validity":
            continue
        for i, m in enumerate(METHODS):
            for proto, dy, filled in [("in-sample", -0.13, True), ("LOFO (mean of folds)", 0.13, False)]:
                r = methods[(methods.method == m) & (methods.protocol == proto) & (methods.metric == metric)]
                if r.empty or np.isnan(r["mean"].iloc[0]):
                    continue
                r = r.iloc[0]
                ax.plot([r.lo, r.hi], [i + dy] * 2, color=COLOR[m], lw=1.2, solid_capstyle="round")
                ax.plot(r["mean"], i + dy, "o", ms=4.5, mfc=COLOR[m] if filled else SURFACE, mec=COLOR[m], mew=1.2)
        ax.set_title(NICE[metric], fontsize=7.5, color=INK, loc="left")
        ax.set_xlim(0, 1.02)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        grid(ax)
    axes[0].set_yticks(range(len(METHODS)), [LABEL[m] for m in METHODS])
    axes[0].invert_yaxis()
    h = [plt.Line2D([], [], marker="o", ls="", ms=4.5, mfc=INK2, mec=INK2, label="in-sample (all 15 models)"),
         plt.Line2D([], [], marker="o", ls="", ms=4.5, mfc=SURFACE, mec=INK2, label="held-out family (LOFO)")]
    fig.legend(handles=h, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.98), fontsize=7)
    save(fig, "fig2_stability")


def fig_tradeoff(methods):
    """Robustness bought vs distance paid: LOFO validity and in-sample feature-set consistency vs proximity."""
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.1))
    pick = lambda m, p, k: methods[(methods.method == m) & (methods.protocol == p) & (methods.metric == k)].iloc[0]
    for ax, (ykey, yproto, ylab) in zip(axes, [("cross_model_validity", "LOFO (mean of folds)", "Validity on held-out family"),
                                              ("feature_set_consistency", "in-sample", "Feature-set consistency")]):
        for m in METHODS:
            x, y = pick(m, "in-sample", "proximity"), pick(m, yproto, ykey)
            if np.isnan(x["mean"]) or np.isnan(y["mean"]):
                continue
            ax.errorbar(x["mean"], y["mean"], xerr=[[x["mean"] - x.lo], [x.hi - x["mean"]]],
                        yerr=[[y["mean"] - y.lo], [y.hi - y["mean"]]], fmt="o", ms=5, color=COLOR[m], elinewidth=1,
                        capsize=0, mec=SURFACE, mew=1.5)
            off, ha = {"DiCE": ((6, 4), "left"), CONS: ((-6, 6), "right"), "ElliCE": ((0, -22), "center")}[m]
            ax.annotate(f"{LABEL[m]} ({y['mean']:.2f})", (x["mean"], y["mean"]), xytext=off, ha=ha,
                        textcoords="offset points", fontsize=6.8, color=INK)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1.05)
        lo_x, hi_x = ax.get_xlim()
        ax.set_xlim(lo_x - 0.1, hi_x + 0.1)
        grid(ax, "both")
    fig.supxlabel("Proximity: L1 change, range-normalised (lower = closer)", fontsize=7.5, color=INK2, y=-0.04)
    save(fig, "fig3_tradeoff")


def fig_features(features):
    """Which actionable feature each method/family changes (share of CFs). Sequential blue, values printed."""
    f = features[features.family != "agreement"]
    f = f[f.method.isin(["DiCE", CONS, "ElliCE"])]
    short = {"DiCE": "DiCE", CONS: "Consensus", "ElliCE": "ElliCE"}  # tau given in the caption
    f = f.assign(col=f.method.map(short) + "\n" + f.family.str.upper())
    order = [f"{short[m]}\n{fam.upper()}" for m in METHODS for fam in ["xgb", "lr", "mlp"]
             if f"{short[m]}\n{fam.upper()}" in set(f.col)]
    M = f.pivot(index="feature", columns="col", values="share_changed").loc[list(NICE)[4:], order]
    fig, ax = plt.subplots(figsize=(7.16, 1.6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])
    im = ax.imshow(M.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(M.values):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.3, color="white" if v > 0.55 else INK)
    ax.set_xticks(range(len(order)), order, fontsize=6.3)
    ax.set_yticks(range(len(M)), [NICE[x] for x in M.index])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("share of CFs changing it", fontsize=6.5)
    cb.outline.set_visible(False)
    save(fig, "fig4_features")


def fig_coverage(coverage, eps):
    """(a) recourse that exists (gray outline) vs what each method finds, per family; (b) ElliCE coverage vs eps."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.0), gridspec_kw={"width_ratios": [1.6, 1]})
    fams = ["xgb", "lr", "mlp", "all"]
    series = ["exists (grid upper bound)"] + METHODS
    w = 0.19
    for k, s in enumerate(series):
        for i, fam in enumerate(fams):
            r = coverage[(coverage.family == fam) & (coverage.method == s)]
            if r.empty:
                continue
            r = r.iloc[0]
            x = i + (k - 1.5) * (w + 0.02)
            if k == 0:
                a.bar(x, r["mean"], w, fc=SURFACE, ec=MUTED, lw=0.9)
            else:
                a.bar(x, r["mean"], w, color=COLOR[s])
            if r["mean"] >= 0.005:  # zero bars stay unlabeled (values in Table 4)
                a.text(x, r["mean"] + 0.02, f"{r['mean']:.2f}".lstrip("0"), ha="center", fontsize=5.2, color=INK2)
    a.set_xticks(range(len(fams)), ["XGBoost", "Logistic reg.", "MLP", "All 15"])
    a.set_ylabel("Share of (patient, model) pairs\nwith a counterfactual")
    a.set_ylim(0, 1.1)
    grid(a, "y")
    h = [matplotlib.patches.Patch(fc=SURFACE, ec=MUTED, label="exists (exhaustive grid)")] + \
        [matplotlib.patches.Patch(color=COLOR[m], label=LABEL[m]) for m in METHODS]
    a.legend(handles=h, fontsize=6.3, ncol=2, loc="upper right")
    for fam, ls in [("mlp", "-"), ("lr", "--")]:
        e = eps[eps.family == fam].sort_values("eps")
        b.plot(e.eps, e["mean"], ls, color=COLOR["ElliCE"], lw=2, marker="o", ms=4)
        b.annotate(fam.upper(), (e.eps.iloc[0], e["mean"].iloc[0]), xytext=(6, 0), textcoords="offset points",
                   fontsize=7, color=INK)
    b.set_xscale("log")
    b.set_xlabel("ElliCE ε (Rashomon loss tolerance)")
    b.set_ylabel("ElliCE coverage\n(stability cohort)")
    b.set_ylim(0, 1.05)
    grid(b, "both")
    save(fig, "fig5_coverage")


def main():
    methods = pd.read_csv(EVAL / "methods.csv")
    fig_pipeline()
    fig_stability(methods)
    fig_tradeoff(methods)
    fig_features(pd.read_csv(EVAL / "features.csv"))
    fig_coverage(pd.read_csv(EVAL / "coverage.csv"), pd.read_csv(EVAL / "ellice_eps.csv"))
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
