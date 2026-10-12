"""Plot the NHANES covariate-specific AUC bands produced by local_knn_bands_real.py.

Two figures per neighbourhood size k:

  nhanes_bands_covariate_k{K}.png
      The covariate-space neighbourhood: the local rank-based estimate with its
      pointwise 95% interval, and the FNN plug-in estimate overlaid, so that the
      points where the plug-in falls outside the interval are visible.

  nhanes_bands_three_k{K}.png
      The same panels with all three neighbourhood representations
      (covariate, model, residualised+covariate) on one set of axes.

Three further figures read the whole k sweep at once and do not depend on k:

  nhanes_ksweep_operating_point.png
      The two errors that trade off against each other as k grows, on separate
      axes because they are measured in different units: the width of the
      interval, which falls as 1/sqrt(k), and the distance between the
      neighbourhood's median age and the age it is meant to describe, which
      rises. Choosing k means choosing a point on this pair of curves.

  nhanes_ksweep_age_slope.png
      The AUC trend in age, per decade, as a function of k, with a two-standard-
      error ribbon, computed over the whole grid and again over ages 40+. Where
      the two disagree, or where the trend changes sign as k grows, the trend is
      an artefact of the neighbourhood rather than a feature of the cohort.

  nhanes_bands_k_contrast.png
      One panel per horizon/sex with three neighbourhood sizes overlaid, so the
      flattening of the curve and the narrowing of the interval are visible side
      by side rather than across figures.

Run from the repository root:

    python src/real_data/plot_nhanes_bands.py            # k = 50
    python src/real_data/plot_nhanes_bands.py -k 20 30 50
    python src/real_data/plot_nhanes_bands.py -i output/knn_bands_real_ksweep \
        -o output/knn_bands_real_ksweep -k 20 30 50 --sweep
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
from matplotlib.transforms import blended_transform_factory
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import pandas as pd

HORIZONS = ["3-year", "5-year", "8-year"]
SEXES = ["female", "male"]

# Categorical slots 1-3 of the reference palette, which are the three that clear the
# all-pairs colour-vision-deficiency floor; the aqua sits below 3:1 against white, so
# every figure using it also carries direct labels. Sex is carried by line style, not
# by colour, so identity never rests on hue alone.
HUE = {"3-year": "#2a78d6", "5-year": "#eb6834", "8-year": "#1baf7a"}

# Fixed meanings, so a colour carries the same entity in every figure here. Orange is the
# network and nothing else: it was also serving as "survey-weighted" in the two weight
# figures, and a reader moving between them read the weighted k-NN curve as the network.
ROLE = {"knn": HUE["3-year"],        # the local rank-based estimate
        "plugin": HUE["5-year"],     # the FNN plug-in
        "weighted": HUE["8-year"]}   # the same estimate, survey-weighted
DASH = {"female": "-", "male": "--"}
INK, INK2 = "#0b0b0b", "#52514e"
AGE_FLOOR = 40.0        # below this the neighbourhood stops being local (R2's MC7)
REPS = [("covariate", "output/knn_bands_real", "#4c72b0"),
        ("model", "output/knn_real_model", "#dd8452"),
        ("residualised + covariate", "output/knn_real_resid", "#55a868")]
PRIMARY = REPS[0][1]


def set_primary(path):
    """Point the covariate panel at a different run (e.g. a re-run with more k)."""
    global PRIMARY, REPS
    PRIMARY = path
    REPS = [(REPS[0][0], path, REPS[0][2])] + REPS[1:]


def _frame(ax, i, j, title):
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0.3, 1.0)
    if i == 1:
        ax.set_xlabel("Age (years)")
    if j == 0:
        ax.set_ylabel("AUC")


def figure_covariate(k, out_dir, k0=None):
    d = _one_k0(pd.read_csv(f"{PRIMARY}/bands_all.csv"), k0)
    s = d[d.k == k]
    if s.empty:
        print(f"  k={k}: no rows in {PRIMARY}/bands_all.csv -- skipped")
        return None

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for i, sx in enumerate(SEXES):
        for j, h in enumerate(HORIZONS):
            ax = axes[i, j]
            g = s[(s.Sex == sx) & (s.Horizon == h)].sort_values("Age")
            if g.empty:
                # A combination can be missing this k even when others have it: k is
                # clipped per arm, so a symmetric request lands on an asymmetric pair
                # wherever the case arm is smaller than k.
                _frame(ax, i, j, f"{sx} · {h}   (no k={k})")
                continue
            ax.axhline(0.5, color="0.7", lw=0.8, ls=":")
            ax.fill_between(g.Age, g["CI95 Lower"], g["CI95 Upper"],
                            color="#4c72b0", alpha=0.18, lw=0, label="95% CI (local)")
            ax.plot(g.Age, g.AUC, color="#4c72b0", lw=2,
                    label="local k-NN (rank-based)")
            ax.plot(g.Age, g["AUC Parametric"], color="#c44e52", lw=2, ls="--",
                    label="FNN plug-in")
            out = ((g["AUC Parametric"] < g["CI95 Lower"])
                   | (g["AUC Parametric"] > g["CI95 Upper"]))
            ax.plot(g.Age[out], g["AUC Parametric"][out], "o", color="#c44e52",
                    ms=5, mfc="white", mew=1.6, label="FNN outside CI")
            n_cases = int(g["N Cases"].iloc[0])
            _frame(ax, i, j,
                   f"{sx} · {h}   ({100 * (~out).mean():.0f}% inside, "
                   f"{n_cases} cases)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("NHANES: covariate-specific AUC, covariate-space neighbourhood "
                 f"(k={k})", fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    path = os.path.join(out_dir, f"nhanes_bands_covariate_k{k}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def figure_three(k, out_dir, k0=None):
    frames = {}
    for name, path, col in REPS:
        f = f"{path}/bands_all.csv"
        if os.path.exists(f):
            frames[name] = _one_k0(pd.read_csv(f), k0)
    base = frames.get("covariate")
    if base is None or base[base.k == k].empty:
        print(f"  k={k}: nothing to plot for the three-representation figure")
        return None

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for i, sx in enumerate(SEXES):
        for j, h in enumerate(HORIZONS):
            ax = axes[i, j]
            ax.axhline(0.5, color="0.7", lw=0.8, ls=":")
            for name, _, col in REPS:
                dd = frames.get(name)
                if dd is None:
                    continue
                g = dd[(dd.k == k) & (dd.Sex == sx)
                       & (dd.Horizon == h)].sort_values("Age")
                if g.empty:
                    continue
                ax.fill_between(g.Age, g["CI95 Lower"], g["CI95 Upper"],
                                color=col, alpha=0.12, lw=0)
                ax.plot(g.Age, g.AUC, color=col, lw=1.8, label=name)
            g0 = base[(base.k == k) & (base.Sex == sx)
                      & (base.Horizon == h)].sort_values("Age")
            ax.plot(g0.Age, g0["AUC Parametric"], color="black", lw=1.6, ls="--",
                    label="FNN plug-in")
            _frame(ax, i, j, f"{sx} · {h}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(f"NHANES: the three neighbourhood representations (k={k})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    path = os.path.join(out_dir, f"nhanes_bands_three_k{k}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _one_k0(d, k0=None):
    """Keep one control-arm size, so a frame has exactly one row per (combo, age, k).

    The symmetric runs carry no "k Controls" column and need no gate. An asymmetric run
    holds several control-arm sizes against the same case-arm k, and every figure that
    selects on `k` alone would silently stack them on top of each other; k0=None keeps
    the symmetric rows, which is the like-for-like default.
    """
    if "k Controls" not in d:
        return d
    return d[d["k Controls"] == k0] if k0 is not None \
        else d[d["k Controls"] == d["k Cases"]]


def _recessive(ax):
    """Grid and spines below the data, text in ink rather than in a series colour."""
    ax.grid(True, color="0.90", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("0.75")
    ax.tick_params(colors=INK2, labelsize=8.5)


def _age_slope(g, floor=None):
    """AUC change per decade of age, weighted by 1/SE^2, with its standard error.

    Weighted because the per-point standard error varies over the grid: the saturated
    cells carry almost no information and would otherwise pull the line.
    """
    if floor is not None:
        g = g[g.Age >= floor]
    if len(g) < 3:
        return np.nan, np.nan
    a = g.Age.to_numpy(float)
    A = np.column_stack([np.ones_like(a), (a - a.mean()) / 10.0])
    w = 1.0 / g.SE.to_numpy(float) ** 2
    beta = np.linalg.lstsq(A * np.sqrt(w)[:, None],
                           g.AUC.to_numpy(float) * np.sqrt(w), rcond=None)[0]
    se = np.linalg.inv((A * w[:, None]).T @ A)[1, 1] ** 0.5
    return float(beta[1]), float(se)


def figure_operating_point(out_dir, k0=None):
    """Interval width and loss of locality, both against k.

    Two panels rather than two y-axes on one: the quantities are in different units
    (AUC and years) and a shared axis would imply a crossing point that depends only
    on the scaling chosen.
    """
    d = _one_k0(pd.read_csv(f"{PRIMARY}/bands_all.csv"), k0)
    d = d.assign(hw=(d["CI95 Upper"] - d["CI95 Lower"]) / 2)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # Left: the half-width is a function of k alone -- the DeLong variance is built
    # from k observations per arm, and the 2,200-2,500 controls on hand never enter.
    # Drawing six near-coincident curves would imply a difference that is not there,
    # so the six are pooled into one line with the spread across them as an envelope.
    ax = axes[0]
    by_k = d.groupby(["k", "Horizon", "Sex"]).hw.mean().groupby("k")
    lo, mid, hi = by_k.min(), by_k.mean(), by_k.max()
    ax.fill_between(mid.index, lo.values, hi.values, color="0.80", lw=0,
                    label="range over the six combinations")
    ax.plot(mid.index, mid.values, color=INK, lw=1.8, marker="o", ms=4.5,
            mec="white", mew=0.9, label="mean over the six combinations")
    for k in (10, 30, 100):
        if k in mid.index:
            ax.annotate(f"  {mid[k]:.3f}", (k, mid[k]), color=INK2, fontsize=8,
                        va="bottom")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK2)
    ax.set_title("Sampling error: set by $k$ alone, falls as $1/\\sqrt{k}$",
                 fontsize=10, color=INK)
    ax.set_ylabel("Mean half-width of the 95% interval (AUC)", color=INK2, fontsize=9)

    # Right: the locality cost does differ by combination, because it is set by how
    # thinly the cases are spread over the age range.
    ax = axes[1]
    for h in HORIZONS:
        for sx in SEXES:
            g = (d[(d.Horizon == h) & (d.Sex == sx)]
                 .groupby("k")["Age Centre Error"].mean().sort_index())
            ax.plot(g.index, g.values, DASH[sx], color=HUE[h], lw=1.8,
                    marker="o", ms=4.5, mec="white", mew=0.9, label=f"{h} {sx}")
            ax.annotate(f"  {h[0]}y {sx[0].upper()}", (g.index[-1], g.values[-1]),
                        color=HUE[h], fontsize=7.5, va="center")
    ax.set_title("Locality error: differs by combination, rises with $k$",
                 fontsize=10, color=INK)
    ax.set_ylabel("Mean |median neighbour age $-$ target age| (years)",
                  color=INK2, fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK2, ncol=2)

    # The scale has to be set before the limits: fixing a limit on a linear axis and
    # then switching to log leaves the other end wherever autoscale had put it.
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(4.2, 420)
        ax.set_xticks([5, 10, 20, 50, 100, 200])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("$k$ (neighbours per arm, log scale)", color=INK2, fontsize=9)
        ax.set_ylim(bottom=0)
        _recessive(ax)

    fig.suptitle("NHANES: choosing the neighbourhood size", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(out_dir, "nhanes_ksweep_operating_point.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_age_slope(out_dir, k0=None):
    """The age trend against k, over the whole grid and over ages 40+."""
    d = _one_k0(pd.read_csv(f"{PRIMARY}/bands_all.csv"), k0)
    series = [("whole grid", None, HUE["3-year"]),
              (f"ages {AGE_FLOOR:.0f}+", AGE_FLOOR, HUE["5-year"])]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for i, sx in enumerate(SEXES):
        for j, h in enumerate(HORIZONS):
            ax = axes[i, j]
            ax.axhline(0.0, color="0.55", lw=0.9)
            s = d[(d.Sex == sx) & (d.Horizon == h)]
            for name, floor, col in series:
                ks, bs, ses = [], [], []
                for k, g in s.groupby("k"):
                    b, se = _age_slope(g.sort_values("Age"), floor)
                    if np.isfinite(b):
                        ks.append(k); bs.append(b); ses.append(se)
                ks, bs, ses = np.array(ks), np.array(bs), np.array(ses)
                ax.fill_between(ks, bs - 2 * ses, bs + 2 * ses, color=col,
                                alpha=0.15, lw=0)
                ax.plot(ks, bs, color=col, lw=1.8, marker="o", ms=4.5,
                        mec="white", mew=0.9, label=f"{name} ($\\pm$2 s.e.)")
            ax.set_xscale("log")
            ax.set_xlim(4.2, 300)
            ax.set_xticks([5, 10, 20, 50, 100, 200])
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.set_title(f"{sx} · {h}   ({int(s['N Cases'].iloc[0])} cases)",
                         fontsize=10, color=INK)
            if i == 1:
                ax.set_xlabel("$k$ (log scale)", color=INK2, fontsize=9)
            if j == 0:
                ax.set_ylabel("AUC change per decade of age", color=INK2, fontsize=9)
            _recessive(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=9, labelcolor=INK2)
    fig.suptitle("NHANES: the age trend is an artefact of the neighbourhood where it "
                 "changes sign with $k$", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    path = os.path.join(out_dir, "nhanes_ksweep_age_slope.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_k_contrast(out_dir, small=10, chosen=30, k0=None):
    """Three neighbourhood sizes per panel: small, the chosen one, the whole case arm.

    The largest k available in each combination is the whole case arm, where the case
    neighbourhood no longer depends on the target age at all; it is drawn to show what
    the band converges to, not as a candidate.
    """
    d = _one_k0(pd.read_csv(f"{PRIMARY}/bands_all.csv"), k0)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for i, sx in enumerate(SEXES):
        for j, h in enumerate(HORIZONS):
            ax = axes[i, j]
            s = d[(d.Sex == sx) & (d.Horizon == h)]
            kmax = int(s.k.max())
            # The whole-arm k differs from panel to panel (it IS the case count), so
            # its size is written beside its own curve and the shared legend names it
            # only by what it is.
            picks = [(small, HUE["3-year"]), (chosen, HUE["5-year"]),
                     (kmax, HUE["8-year"])]
            ax.axhline(0.5, color="0.7", lw=0.8, ls=":")
            ax.axvline(AGE_FLOOR, color="0.55", lw=1.0, ls=(0, (4, 3)))
            ends = []
            for k, col in picks:
                g = s[s.k == k].sort_values("Age")
                if g.empty:
                    continue
                # Only the chosen k carries its interval. Three translucent ribbons
                # over one another mix into a colour none of them is, and the widest
                # of them belongs to the size being argued against.
                if k == chosen:
                    ax.fill_between(g.Age, g["CI95 Lower"], g["CI95 Upper"],
                                    color=col, alpha=0.18, lw=0)
                ax.plot(g.Age, g.AUC, color=col, lw=1.9,
                        path_effects=[pe.Stroke(linewidth=3.4, foreground="white"),
                                      pe.Normal()])
                ends.append([float(g.Age.iloc[-1]), float(g.AUC.iloc[-1]), k, col])
            # The curves converge at the right edge, so the end labels have to be
            # pushed apart or they overprint one another.
            ends.sort(key=lambda e: e[1])
            for a, b in zip(ends, ends[1:]):
                b[1] = max(b[1], a[1] + 0.045)
            for x, y, k, col in ends:
                ax.annotate(f"  {k}", (x, y), color=col, fontsize=8, va="center",
                            fontweight="bold")
            ax.set_title(f"{sx} · {h}   ({int(s['N Cases'].iloc[0])} cases)",
                         fontsize=10, color=INK)
            ax.set_ylim(0.2, 1.0)
            ax.set_xlim(right=88)
            if i == 1:
                ax.set_xlabel("Age (years)", color=INK2, fontsize=9)
            if j == 0:
                ax.set_ylabel("AUC", color=INK2, fontsize=9)
            _recessive(ax)

    # The largest k drawn is the whole case arm only when the run actually reaches it;
    # under a k0 filter it is merely the largest case arm the filter left behind, and
    # calling it "the whole arm" would be a false label.
    whole = all(int(g.k.max()) == int(g["N Cases"].iloc[0])
                for _, g in d.groupby(["Horizon", "Sex"]))
    handles = [Line2D([], [], color=HUE["3-year"], lw=1.9),
               Line2D([], [], color=HUE["5-year"], lw=1.9),
               Line2D([], [], color=HUE["8-year"], lw=1.9),
               Line2D([], [], color="0.55", lw=1.0, ls=(0, (4, 3)))]
    labels = [f"$k={small}$", f"$k={chosen}$, with its 95% interval",
              "$k=$ the whole case arm (size beside each curve)" if whole
              else "$k=$ the largest case arm in this run (size beside each curve)",
              f"age {AGE_FLOOR:.0f}: below it the neighbourhood is not local"]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=9, labelcolor=INK2)
    fig.suptitle("NHANES: the band at three neighbourhood sizes", fontsize=12,
                 color=INK)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    path = os.path.join(out_dir, "nhanes_bands_k_contrast.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_weight_contrast(out_dir, other_dir, k, k0=None):
    """The same band with and without the survey weights, at one k.

    PRIMARY is the weighted run and `other_dir` the unweighted one, or the reverse --
    the figure only needs the two to differ in that one respect. The two estimate
    different things: weighted, the AUC of the population the survey represents;
    unweighted, the AUC of the cohort as sampled.
    """
    a = _one_k0(pd.read_csv(f"{other_dir}/bands_all.csv"), k0)
    b = _one_k0(pd.read_csv(f"{PRIMARY}/bands_all.csv"), k0)
    if b[b.k == k].empty or a[a.k == k].empty:
        print(f"  k={k}: missing from one of the two runs -- skipped")
        return None

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for i, sx in enumerate(SEXES):
        for j, h in enumerate(HORIZONS):
            ax = axes[i, j]
            ax.axhline(0.5, color="0.7", lw=0.8, ls=":")
            ax.axvline(AGE_FLOOR, color="0.55", lw=1.0, ls=(0, (4, 3)))
            for dd, col, lab in ((a, ROLE["knn"], "local k-NN, unweighted"),
                                 (b, ROLE["weighted"], "local k-NN, survey-weighted")):
                g = dd[(dd.k == k) & (dd.Sex == sx)
                       & (dd.Horizon == h)].sort_values("Age")
                ax.fill_between(g.Age, g["CI95 Lower"], g["CI95 Upper"], color=col,
                                alpha=0.15, lw=0)
                ax.plot(g.Age, g.AUC, color=col, lw=1.9, label=lab,
                        path_effects=[pe.Stroke(linewidth=3.4, foreground="white"),
                                      pe.Normal()])
            gb = b[(b.k == k) & (b.Sex == sx) & (b.Horizon == h)]
            # How much of the neighbourhood survives the weighting, in Kish's sense.
            neff = (gb["N1 Eff"].mean() if "N1 Eff" in gb else float("nan"))
            _frame(ax, i, j, f"{sx} · {h}   ($k={k}$ cases $\\to$ "
                             f"{neff:.0f} effective)")
            ax.set_ylim(0.2, 1.0)
            _recessive(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.55", lw=1.0, ls=(0, (4, 3))))
    labels.append(f"age {AGE_FLOOR:.0f}")
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=9, labelcolor=INK2)
    fig.suptitle("NHANES: the effect of the survey weights on the band "
                 f"($k={k}$, 95% interval)", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    path = os.path.join(out_dir, f"nhanes_weights_effect_k{k}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_realized_age(out_dir, k0, k1):
    """Where the grid actually is: the nominal age against the case arm's median age.

    A grid point is a label, not a fact. Where the curve leaves the diagonal, the cases
    the estimate is built from are not at the age the axis claims, and no neighbourhood
    size repairs it -- the cases are simply not there.
    """
    d = pd.read_csv(f"{PRIMARY}/bands_all.csv")
    s = d[(d["k Controls"] == k0) & (d["k Cases"] == k1)]
    if s.empty:
        print(f"  {k0}:{k1} not in {PRIMARY} -- skipped")
        return None

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    lo, hi = s.Age.min(), s.Age.max()
    ax.plot([lo, hi], [lo, hi], color="0.55", lw=1.2, ls=(0, (4, 3)),
            label="the grid point is honest", zorder=1)
    for h in HORIZONS:
        for sx in SEXES:
            g = s[(s.Horizon == h) & (s.Sex == sx)].sort_values("Age")
            ax.plot(g.Age, g["Case Median Age"], DASH[sx], color=HUE[h], lw=1.8,
                    marker="o", ms=4.5, mec="white", mew=0.9, label=f"{h} {sx}",
                    zorder=2)
    ax.set_xlabel("Nominal age of the grid point (years)", color=INK2, fontsize=10)
    ax.set_ylabel(f"Median age of the $k={k1}$ cases actually used (years)",
                  color=INK2, fontsize=10)
    ax.set_title("NHANES: the grid point and the cases behind it", fontsize=12,
                 color=INK)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)
    _recessive(ax)
    fig.tight_layout()
    path = os.path.join(out_dir, "nhanes_realized_age_map.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_matched_arms(out_dir, other_dir, k0, k1):
    """AUC against the age the estimate really describes, controls matched and not.

    PRIMARY is the matched run, `other_dir` the unmatched one. Both select the same
    cases, so both are drawn at the same realized age and the only difference is where
    the controls came from.
    """
    def pick(path):
        x = pd.read_csv(f"{path}/bands_all.csv")
        return x[(x["k Controls"] == k0) & (x["k Cases"] == k1)]
    a, b = pick(other_dir), pick(PRIMARY)
    if a.empty or b.empty:
        print(f"  {k0}:{k1} missing from one of the two runs -- skipped")
        return None
    # Matching moves only the controls, so both runs use the very same cases and share
    # one realized age. An older unmatched run may predate the column; take it from the
    # matched one rather than recomputing it.
    if "Case Median Age" not in a:
        a = a.merge(b[["Horizon", "Sex", "Age", "Case Median Age"]],
                    on=["Horizon", "Sex", "Age"], how="left")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for i, sx in enumerate(SEXES):
        for j, h in enumerate(HORIZONS):
            ax = axes[i, j]
            ax.axhline(0.5, color="0.7", lw=0.8, ls=":")
            for dd, col, lab in ((a, HUE["3-year"], "controls at the grid age"),
                                 (b, HUE["5-year"], "controls matched to the cases")):
                g = (dd[(dd.Sex == sx) & (dd.Horizon == h)]
                     .sort_values("Case Median Age"))
                ax.errorbar(g["Case Median Age"], g.AUC,
                            yerr=[g.AUC - g["CI95 Lower"], g["CI95 Upper"] - g.AUC],
                            fmt="o", ms=4.5, lw=1.0, color=col, mec="white", mew=0.8,
                            alpha=0.85, capsize=0, label=lab)
            # Several nominal grid points land on the same realized age; the vertical
            # scatter at one x is exactly that collapse, and is the point of the figure.
            gg = b[(b.Sex == sx) & (b.Horizon == h)]
            span = f"{gg['Case Median Age'].min():.0f}-{gg['Case Median Age'].max():.0f}"
            _frame(ax, i, j, f"{sx} · {h}   (21-80 collapses onto {span})")
            ax.set_xlabel("" if i == 0 else "Median age of the cases used (years)")
            ax.set_ylim(0.2, 1.0)
            _recessive(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=9, labelcolor=INK2)
    fig.suptitle("NHANES: the AUC against the age it actually describes "
                 f"($k_0={k0}$, $k_1={k1}$, 95% interval)", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    path = os.path.join(out_dir, f"nhanes_matched_arms_k{k0}_{k1}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _control_offset(data_dir, k0_values):
    """Mean |median control age - target age| per (sex, horizon, k0).

    Not in the CSV, but it needs nothing that is: the neighbourhood of the control arm
    is a deterministic function of the covariates and k0, with no fitted model in it, so
    it is rebuilt here exactly as run_one forms it.
    """
    cov, age = ["RIDAGEYR.x", "BMI"], "RIDAGEYR.x"
    out = {}
    for sx, fn in (("female", "df_f.csv"), ("male", "df_m.csv")):
        for col, hname in (("tres", "3-year"), ("cinco", "5-year"), ("ocho", "8-year")):
            df = (pd.read_csv(os.path.join(data_dir, fn))
                  .dropna(subset=cov + ["TAC2", col]).reset_index(drop=True))
            i0 = np.where(df[col].to_numpy() == 0)[0]
            scv = np.array([df[c].std() for c in cov], float)
            ages = np.linspace(df[age].quantile(.025), df[age].quantile(.975), 21)
            raw = np.column_stack([ages, np.full(21, df[cov[1]].median())])
            c0 = np.column_stack([df[c].to_numpy(float)[i0] for c in cov]) / scv
            a0 = df[age].to_numpy()[i0]
            for k0 in k0_values:
                if k0 > len(i0):
                    continue
                off = [abs(np.median(a0[np.argsort(
                    np.linalg.norm(c0 - raw[t] / scv, axis=1))[:k0]]) - raw[t, 0])
                    for t in range(21)]
                out[(sx, hname, k0)] = float(np.mean(off))
    return out


def figure_control_arm(out_dir, k1, data_dir):
    """What enlarging the control arm buys and what it costs, at a fixed case arm.

    Three panels because there are three quantities in three units; the case arm is held
    at k1 throughout, so its own locality is constant and only the control arm moves.
    """
    d = pd.read_csv(f"{PRIMARY}/bands_all.csv")
    d = d[d["k Cases"] == k1].assign(hw=(d["CI95 Upper"] - d["CI95 Lower"]) / 2)
    if d.empty:
        print(f"  no rows with k Cases == {k1} in {PRIMARY} -- skipped")
        return None
    off = _control_offset(data_dir, sorted(d["k Controls"].unique()))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for h in HORIZONS:
        for sx in SEXES:
            g = d[(d.Horizon == h) & (d.Sex == sx)]
            ks = sorted(g["k Controls"].unique())
            style = dict(color=HUE[h], lw=1.8, marker="o", ms=4.5, mec="white", mew=0.9)
            axes[0].plot(ks, [g[g["k Controls"] == k].hw.mean() for k in ks],
                         DASH[sx], label=f"{h} {sx}", **style)
            axes[1].plot(ks, [off.get((sx, h, k), np.nan) for k in ks], DASH[sx], **style)
            axes[2].plot(ks, [_age_slope(g[g["k Controls"] == k].sort_values("Age"))[0]
                              for k in ks], DASH[sx], **style)
    titles = ["Interval width: what it buys",
              "Control locality: what it costs",
              "Age trend: where the cost shows"]
    ylabs = ["Mean half-width of the 95% interval (AUC)",
             "Mean |median control age $-$ target age| (years)",
             "AUC change per decade of age"]
    for ax, ti, yl in zip(axes, titles, ylabs):
        ax.set_xscale("log")
        ax.set_xticks([10, 20, 50, 100, 200, 500, 1000, 2000])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("$k_0$, controls per neighbourhood (log scale)", color=INK2,
                      fontsize=9)
        ax.set_ylabel(yl, color=INK2, fontsize=9)
        ax.set_title(ti, fontsize=10, color=INK)
        _recessive(ax)
    axes[2].axhline(0.0, color="0.55", lw=0.9)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    axes[0].legend(loc="upper right", frameon=False, fontsize=8, labelcolor=INK2, ncol=2)
    fig.suptitle(f"NHANES: sizing the control arm against a fixed case arm ($k_1={k1}$)",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out_dir, f"nhanes_control_arm_k1_{k1}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_k0_contrast(out_dir, k1, picks=(None, 200, 1000)):
    """The band at a fixed case arm and three control arms.

    picks[0] of None means the symmetric k0 = k1, the starting point.
    """
    d = pd.read_csv(f"{PRIMARY}/bands_all.csv")
    d = d[d["k Cases"] == k1]
    if d.empty:
        print(f"  no rows with k Cases == {k1} in {PRIMARY} -- skipped")
        return None
    cols = [HUE["3-year"], HUE["5-year"], HUE["8-year"]]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for i, sx in enumerate(SEXES):
        for j, h in enumerate(HORIZONS):
            ax = axes[i, j]
            s = d[(d.Sex == sx) & (d.Horizon == h)]
            ax.axhline(0.5, color="0.7", lw=0.8, ls=":")
            ax.axvline(AGE_FLOOR, color="0.55", lw=1.0, ls=(0, (4, 3)))
            for pick, col in zip(picks, cols):
                k0 = k1 if pick is None else pick
                g = s[s["k Controls"] == k0].sort_values("Age")
                if g.empty:
                    continue
                # Only the middle choice carries its interval: three overlapping ribbons
                # mix into a colour none of them is.
                if pick == picks[1]:
                    ax.fill_between(g.Age, g["CI95 Lower"], g["CI95 Upper"], color=col,
                                    alpha=0.18, lw=0)
                ax.plot(g.Age, g.AUC, color=col, lw=1.9,
                        path_effects=[pe.Stroke(linewidth=3.4, foreground="white"),
                                      pe.Normal()])
            _frame(ax, i, j, f"{sx} · {h}   ({int(s['N Cases'].iloc[0])} cases)")
            ax.set_ylim(0.2, 1.0)
            _recessive(ax)

    handles = [Line2D([], [], color=c, lw=1.9) for c in cols]
    labels = [f"$k_0={k1}$ (symmetric)", f"$k_0={picks[1]}$, with its 95% interval",
              f"$k_0={picks[2]}$"]
    handles.append(Line2D([], [], color="0.55", lw=1.0, ls=(0, (4, 3))))
    labels.append(f"age {AGE_FLOOR:.0f}")
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9,
               labelcolor=INK2)
    fig.suptitle(f"NHANES: the band as the control arm grows ($k_1={k1}$ cases "
                 f"throughout)", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    path = os.path.join(out_dir, f"nhanes_k0_contrast_k1_{k1}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _age_rule(ax, g, floor):
    """Draw the locality cut-off, but only where it falls inside the panel.

    A run reported over 50-80 has already excluded the region the rule marks, so drawing
    it there would stretch the axis to an age the figure has nothing to say about.
    """
    if floor is None:
        return False
    lo, hi = float(g.Age.min()), float(g.Age.max())
    if not (lo < floor < hi):
        return False
    ax.axvline(floor, color="0.55", lw=1.0, ls=(0, (4, 3)))
    return True


def _draw_group_panel(ax, g, plugin, floor):
    """One band panel. Shared by the grid and the one-file-per-panel version."""
    ax.axhline(0.5, color="0.7", lw=0.8, ls=":")
    drew = _age_rule(ax, g, floor)
    ax.fill_between(g.Age, g["CI95 Lower"], g["CI95 Upper"], color=ROLE["knn"],
                    alpha=0.18, lw=0, label="95% interval")
    ax.plot(g.Age, g.AUC, color=ROLE["knn"], lw=2.0, label="local k-NN",
            path_effects=[pe.Stroke(linewidth=3.5, foreground="white"), pe.Normal()])
    if plugin:
        ax.plot(g.Age, g["AUC Parametric"], color=ROLE["plugin"], lw=1.8,
                ls=(0, (5, 2)), label="FNN plug-in")
        out = ((g["AUC Parametric"] < g["CI95 Lower"])
               | (g["AUC Parametric"] > g["CI95 Upper"]))
        ax.plot(g.Age[out], g["AUC Parametric"][out], "o", color=ROLE["plugin"], ms=5,
                mfc="white", mew=1.6, label="plug-in outside the interval")
    return drew


def figure_per_group_split(out_dir, plugin=True, floor=AGE_FLOOR):
    """The band, one file per horizon/sex instead of a 2x3 grid.

    The y-axis is held common across the six, as in the grid: read one at a time, panels
    scaled to their own range would make a narrow interval and a wide one look alike.
    """
    d = pd.read_csv(f"{PRIMARY}/bands_all.csv")
    lo_y = float(min(d["CI95 Lower"].min(), d.AUC.min(),
                     d["AUC Parametric"].min() if plugin else 1.0))
    hi_y = float(max(d["CI95 Upper"].max(), d.AUC.max(),
                     d["AUC Parametric"].max() if plugin else 0.0))
    pad = max(0.02, 0.05 * (hi_y - lo_y))

    paths = []
    for sx in SEXES:
        for h in HORIZONS:
            g = d[(d.Sex == sx) & (d.Horizon == h)].sort_values("Age")
            if g.empty:
                print(f"  {sx} {h}: no rows -- skipped")
                continue
            pairs = g[["k Controls", "k Cases"]].drop_duplicates()
            if len(pairs) != 1:
                raise ValueError(f"{h}/{sx} holds {len(pairs)} neighbourhood sizes, "
                                 f"expected one:\n{pairs}")
            k0, k1 = int(pairs.iloc[0, 0]), int(pairs.iloc[0, 1])

            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            drew = _draw_group_panel(ax, g, plugin, floor)
            ax.set_xlabel("Age (years)", color=INK2, fontsize=10)
            ax.set_ylabel("AUC", color=INK2, fontsize=10)
            ax.set_ylim(lo_y - pad, hi_y + pad)
            ax.set_title(f"NHANES {sx} · {h}\n{int(g['N Cases'].iloc[0])} cases · "
                         f"$k_0$={k0}, $k_1$={k1}", fontsize=11, color=INK)
            handles, labels = ax.get_legend_handles_labels()
            if drew:
                handles.append(Line2D([], [], color="0.55", lw=1.0, ls=(0, (4, 3))))
                labels.append(f"age {floor:.0f}")
            ax.legend(handles, labels, loc="lower right", frameon=False, fontsize=8,
                      labelcolor=INK2)
            _recessive(ax)
            fig.tight_layout()
            path = os.path.join(out_dir, f"nhanes_bands_{h}_{sx}.png")
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(path)
    return paths


HORIZON_COL = {"3-year": "tres", "5-year": "cinco", "8-year": "ocho"}
COHORT_FILE = {"female": "df_f.csv", "male": "df_m.csv"}


def figure_per_group_bmi(out_dir, data_dir, plugin=True, floor=AGE_FLOOR, under="bmi"):
    """The band, with the cohort it rests on drawn underneath.

    under="bmi" plots each subject at (age, BMI) with the median BMI the grid sits at.
    under="age" drops BMI and plots the subjects as two jittered lanes along the age
    axis, one per arm: the same support question read on the axis the curve is drawn
    against, which is the axis where the cohort actually runs out.

    Every curve in this set is evaluated at the median BMI and plotted against age, which
    leaves two things invisible: where the subjects actually are along the age axis, and
    whether that median slice is representative of them. A strip under each panel shows
    both -- one point per subject in (age, BMI), the arms distinguished, and the median
    BMI the grid is evaluated at drawn across it.

    The two share the age axis and are stacked rather than overlaid. Putting BMI on a
    second y-axis against the AUC would make the apparent relation between them depend on
    how each axis is scaled, which is a choice the figure should not be making.
    """
    d = pd.read_csv(f"{PRIMARY}/bands_all.csv")
    # Five rows, the middle one an empty spacer: a gridspec has one hspace for every
    # gap, and the gap inside an AUC/BMI pair must be small while the gap between the
    # two pairs has to clear the lower pair's titles.
    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(5, 3, height_ratios=[3, 1.15, 0.55, 3, 1.15], hspace=0.10,
                          wspace=0.14)

    lo_y = float(min(d["CI95 Lower"].min(), d.AUC.min()))
    hi_y = float(max(d["CI95 Upper"].max(), d.AUC.max()))
    if plugin:
        lo_y = min(lo_y, float(d["AUC Parametric"].min()))
        hi_y = max(hi_y, float(d["AUC Parametric"].max()))
    pad = max(0.02, 0.05 * (hi_y - lo_y))

    drew_rule = False
    for i, sx in enumerate(SEXES):
        for j, h in enumerate(HORIZONS):
            row = 0 if i == 0 else 3          # row 2 is the spacer
            ax = fig.add_subplot(gs[row, j])
            axb = fig.add_subplot(gs[row + 1, j], sharex=ax)
            g = d[(d.Sex == sx) & (d.Horizon == h)].sort_values("Age")
            if g.empty:
                continue
            pairs = g[["k Controls", "k Cases"]].drop_duplicates()
            if len(pairs) != 1:
                raise ValueError(f"{h}/{sx} holds {len(pairs)} neighbourhood sizes")
            k0, k1 = int(pairs.iloc[0, 0]), int(pairs.iloc[0, 1])

            # No rug here: this figure already shows the cohort, in the strip below.
            drew_rule = _draw_group_panel(ax, g, plugin, floor) or drew_rule
            ax.set_ylim(lo_y - pad, hi_y + pad)
            ax.set_title(f"{sx} · {h}   ({int(g['N Cases'].iloc[0])} cases · "
                         f"$k_0$={k0}, $k_1$={k1})", fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel("AUC", color=INK2, fontsize=9)
            plt.setp(ax.get_xticklabels(), visible=False)
            _recessive(ax)

            # The cohort underneath.
            col = HORIZON_COL[h]
            df = (pd.read_csv(os.path.join(data_dir, COHORT_FILE[sx]))
                  .dropna(subset=["RIDAGEYR.x", "BMI", "TAC2", col]))
            lo_a, hi_a = float(g.Age.min()), float(g.Age.max())
            span = hi_a - lo_a
            df = df[df["RIDAGEYR.x"].between(lo_a - 0.03 * span, hi_a + 0.03 * span)]
            ctrl, case = df[df[col] == 0], df[df[col] == 1]
            if under == "bmi":
                axb.scatter(ctrl["RIDAGEYR.x"], ctrl.BMI, s=3, color="0.72", lw=0,
                            alpha=0.45, label=f"controls ({len(ctrl)})")
                axb.scatter(case["RIDAGEYR.x"], case.BMI, s=13, facecolors="none",
                            edgecolors=INK, lw=0.8, label=f"cases ({len(case)})")
                med = float(df.BMI.median())
                axb.axhline(med, color="0.45", lw=1.0, ls=(0, (4, 3)),
                            label=f"median BMI ({med:.1f}), where the grid sits")
                # Clipped to the inner range: a handful of BMIs above 50 would otherwise
                # flatten the strip into a line.
                axb.set_ylim(float(df.BMI.quantile(.01)),
                             float(df.BMI.quantile(.99)))
                if j == 0:
                    axb.set_ylabel("BMI", color=INK2, fontsize=9)
            else:
                # One lane per arm. The vertical scatter is jitter and carries no
                # meaning; it is there so that subjects at the same age stay countable
                # instead of collapsing onto one mark -- and at the top of the range
                # they do pile up, because NHANES codes every age above 80 as 80.
                rng = np.random.default_rng(0)
                for arm, y0, col, kw in (
                        (ctrl, 1.0, "0.72",
                         dict(s=3, lw=0, alpha=0.40)),
                        (case, 0.0, INK,
                         dict(s=13, facecolors="none", lw=0.8))):
                    lab = ("controls" if y0 else "cases") + f" ({len(arm)})"
                    y = y0 + rng.uniform(-0.3, 0.3, len(arm))
                    if y0:
                        axb.scatter(arm["RIDAGEYR.x"], y, color=col, label=lab, **kw)
                    else:
                        axb.scatter(arm["RIDAGEYR.x"], y, edgecolors=col, label=lab,
                                    **kw)
                axb.set_ylim(-0.6, 1.6)
                axb.set_yticks([0, 1])
                axb.set_yticklabels(["cases", "controls"], fontsize=8)
                if j == 0:
                    axb.set_ylabel("")
            _age_rule(axb, g, floor)
            axb.set_xlabel("Age (years)", color=INK2, fontsize=9)
            _recessive(axb)

    h1, l1 = fig.axes[0].get_legend_handles_labels()
    h2, l2 = fig.axes[1].get_legend_handles_labels()
    if drew_rule:
        h1.append(Line2D([], [], color="0.55", lw=1.0, ls=(0, (4, 3))))
        l1.append(f"age {floor:.0f}")
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=4, frameon=False,
               fontsize=8.5, labelcolor=INK2)
    fig.suptitle("NHANES: covariate-specific AUC, and the cohort it is computed from",
                 fontsize=12, color=INK)
    # subplots_adjust rather than tight_layout: the latter recomputes the gaps and
    # undoes the spacer row.
    fig.subplots_adjust(left=0.06, right=0.985, top=0.925, bottom=0.10)
    path = os.path.join(out_dir, f"nhanes_bands_per_group_{under}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _cohort_rug(ax, data_dir, horizon, sex, lo_a, hi_a, legend=False):
    """The subjects as two lanes inside the panel, at the foot of the axes.

    Drawn on a blended transform -- age in data coordinates, height as a fraction of the
    axes -- so the lanes sit at the same place in every panel whatever the AUC range,
    and stay put if the limits change. The vertical scatter within a lane is jitter and
    carries no meaning; without it the subjects at a given age collapse onto one mark,
    which at the top of the range is most of them, since NHANES codes every age above 80
    as 80.
    """
    col = HORIZON_COL[horizon]
    df = (pd.read_csv(os.path.join(data_dir, COHORT_FILE[sex]))
          .dropna(subset=["RIDAGEYR.x", "BMI", "TAC2", col]))
    span = hi_a - lo_a
    df = df[df["RIDAGEYR.x"].between(lo_a - 0.03 * span, hi_a + 0.03 * span)]
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    rng = np.random.default_rng(0)
    for arm_val, base, col_, kw in ((0, 0.085, "0.70", dict(s=2.5, lw=0, alpha=0.35)),
                                    (1, 0.028, INK, dict(s=9, lw=0.7))):
        arm = df[df[col] == arm_val]
        lab = (f"controls ({len(arm)})" if arm_val == 0
               else f"cases ({len(arm)})") if legend else None
        y = base + rng.uniform(-0.016, 0.016, len(arm))
        if arm_val == 0:
            ax.scatter(arm["RIDAGEYR.x"], y, transform=tr, color=col_, label=lab,
                       zorder=1, **kw)
        else:
            ax.scatter(arm["RIDAGEYR.x"], y, transform=tr, facecolors="none",
                       edgecolors=col_, label=lab, zorder=2, **kw)


def _draw_subject_panel(ax, g, h, sx, arms, floor):
    """One subject-scatter panel. Shared by the grid and the one-file-per-panel version
    so the two cannot drift apart."""
    sf = os.path.join(PRIMARY, f"subjects_{HORIZON_COL[h]}_{sx}.csv")
    if os.path.exists(sf):
        s = pd.read_csv(sf)
        # Only the subjects the panel actually spans; the fit covers the whole cohort
        # but the axis does not.
        s = s[s["RIDAGEYR.x"].between(g.Age.min(), g.Age.max())]
        if arms:
            for a_, col, lab, kw in (
                    (0, "0.62", "controls", dict(s=7, lw=0, alpha=0.35)),
                    (1, ROLE["plugin"], "cases",
                     dict(s=16, facecolors="none", lw=0.8, alpha=0.8))):
                q = s[s.Arm == a_]
                lab = f"plug-in per subject, {lab} ({len(q)})"
                if a_ == 0:
                    ax.scatter(q["RIDAGEYR.x"], q["AUC Parametric"], color=col,
                               label=lab, zorder=1, **kw)
                else:
                    ax.scatter(q["RIDAGEYR.x"], q["AUC Parametric"], edgecolors=col,
                               label=lab, zorder=2, **kw)
        else:
            ax.scatter(s["RIDAGEYR.x"], s["AUC Parametric"], s=7, color="0.62", lw=0,
                       alpha=0.35, label=f"plug-in per subject ({len(s)})", zorder=1)
    else:
        print(f"  no subjects file for {h}/{sx} -- scatter omitted")

    ax.axhline(0.5, color="0.7", lw=0.8, ls=":", zorder=0)
    drew = _age_rule(ax, g, floor)
    ax.fill_between(g.Age, g["CI95 Lower"], g["CI95 Upper"], color=ROLE["knn"],
                    alpha=0.20, lw=0, label="95% interval", zorder=3)
    ax.plot(g.Age, g.AUC, color=ROLE["knn"], lw=2.2, label="local k-NN", zorder=5,
            path_effects=[pe.Stroke(linewidth=3.8, foreground="white"), pe.Normal()])
    ax.plot(g.Age, g["AUC Parametric"], color=ROLE["plugin"], lw=2.0, ls=(0, (5, 2)),
            label="FNN plug-in, median BMI", zorder=6,
            path_effects=[pe.Stroke(linewidth=3.6, foreground="white"), pe.Normal()])
    return drew


def _prepare_subject_frame(k, clip_half):
    """The band frame the subject figures draw, with the k selection and the optional
    truncation applied once."""
    d = pd.read_csv(f"{PRIMARY}/bands_all.csv")
    if k is not None:
        d = d[d.k == k]
        if d.empty:
            raise SystemExit(f"k={k}: no rows in {PRIMARY}")
    else:
        n = d.groupby(["Horizon", "Sex"]).k.nunique()
        if (n > 1).any():
            raise ValueError("several k per combination; pass a k explicitly\n"
                             f"{n[n > 1]}")
    if clip_half:
        below = d[d.AUC < 0.5]
        if len(below):
            raise ValueError(
                f"{len(below)} point estimates are themselves below 0.5, so the "
                "interval cannot be truncated there without excluding them:\n"
                f"{below[['Horizon', 'Sex', 'Age', 'AUC']].to_string(index=False)}")
        n_clipped = int((d["CI95 Lower"] < 0.5).sum())
        d = d.assign(**{"CI95 Lower": d["CI95 Lower"].clip(lower=0.5)})
        print(f"  interval truncated at 0.5 in {n_clipped} of {len(d)} grid points")
    return d


def figure_subject_scatter_split(out_dir, k, floor=AGE_FLOOR, arms=True,
                                 clip_half=False, k_in_title=True, clip_note=True):
    """The subject-scatter figure, one file per horizon/sex.

    The y-axis is held common across the six, as in the grid: read one at a time, panels
    scaled to their own range would make a narrow interval and a wide one look alike.
    """
    d = _prepare_subject_frame(k, clip_half)
    lo_y, hi_y = 0.3, 1.0
    paths = []
    for sx in SEXES:
        for h in HORIZONS:
            g = d[(d.Sex == sx) & (d.Horizon == h)].sort_values("Age")
            if g.empty:
                print(f"  {sx} {h}: no rows -- skipped")
                continue
            fig, ax = plt.subplots(figsize=(6.6, 5.0))
            drew = _draw_subject_panel(ax, g, h, sx, arms, floor)
            if k_in_title:
                kt = ((int(g["k Controls"].iloc[0]), int(g["k Cases"].iloc[0]))
                      if "k Controls" in g else (None, int(g.k.iloc[0])))
                tt = (f"NHANES {sx} · {h}   ($k_0$={kt[0]}, $k_1$={kt[1]})"
                      if kt[0] is not None else f"NHANES {sx} · {h}   ($k$={kt[1]})")
            else:
                tt = f"NHANES {sx} · {h}"
            if clip_half and clip_note:
                tt += "\ninterval truncated at 0.5"
            ax.set_title(tt, fontsize=11, color=INK)
            ax.set_xlabel("Age (years)", color=INK2, fontsize=10)
            ax.set_ylabel("AUC", color=INK2, fontsize=10)
            ax.set_ylim(lo_y, hi_y)
            handles, labels = ax.get_legend_handles_labels()
            if drew:
                handles.append(Line2D([], [], color="0.55", lw=1.0, ls=(0, (4, 3))))
                labels.append(f"age {floor:.0f}")
            ax.legend(handles, labels, loc="lower left", frameon=False, fontsize=8,
                      labelcolor=INK2)
            _recessive(ax)
            fig.tight_layout()
            path = os.path.join(
                out_dir, "nhanes_bands_subjects"
                + ("_clipped" if clip_half else "") + f"_{h}_{sx}.png")
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(path)
    return paths


def figure_subject_scatter(out_dir, k, floor=AGE_FLOOR, arms=True, clip_half=False,
                           k_in_title=True, clip_note=True):
    """Everything on one set of axes: the band, the rank-based curve, the plug-in curve,
    and one point per subject.

    Each point is the plug-in evaluated at that subject's OWN age and BMI, so it is the
    same estimator as the dashed curve -- the curve is it along the median-BMI slice,
    the cloud is it over the BMI the subjects actually have. The vertical spread at a
    given age is therefore the BMI effect, which the curve hides by construction, and
    the curve should run through the middle of the cloud rather than along its edge.

    clip_half=True truncates the interval at 0.5. That is a statement about the
    parameter space, not a cosmetic one: it says the direction of the marker is known a
    priori -- TAC is protective, so controls lie above cases -- and an interval reaching
    below 0.5 is covering values the design rules out. Truncating to a known parameter
    space does not cost coverage, it only ever adds it. It does hide that the data are
    compatible with no discrimination, so the figure should say the interval is
    truncated. It is refused outright if any point estimate is itself below 0.5, since
    an interval that excludes its own estimate would be nonsense.

    Needs subjects_<horizon>_<sex>.csv, which local_knn_bands_real.py writes under
    --subject_auc.
    """
    d = pd.read_csv(f"{PRIMARY}/bands_all.csv")
    if k is not None:
        d = d[d.k == k]
        if d.empty:
            print(f"  k={k}: no rows in {PRIMARY} -- skipped")
            return None
    else:
        # k=None: take whatever single pair each combination has. A request is clipped
        # per arm, so one combination can hold 78 cases where another holds only the 40
        # it has; filtering on a single number would blank those panels.
        n = d.groupby(["Horizon", "Sex"]).k.nunique()
        if (n > 1).any():
            raise ValueError("several k per combination; pass --subject_scatter K\n"
                             f"{n[n > 1]}")
    if clip_half:
        below = d[d.AUC < 0.5]
        if len(below):
            raise ValueError(
                f"{len(below)} point estimates are themselves below 0.5, so the "
                "interval cannot be truncated there without excluding them:\n"
                f"{below[['Horizon', 'Sex', 'Age', 'AUC']].to_string(index=False)}")
        n_clipped = int((d["CI95 Lower"] < 0.5).sum())
        d = d.assign(**{"CI95 Lower": d["CI95 Lower"].clip(lower=0.5)})
        print(f"  interval truncated at 0.5 in {n_clipped} of {len(d)} grid points")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    drew = False
    for i, sx in enumerate(SEXES):
        for j, h in enumerate(HORIZONS):
            ax = axes[i, j]
            g = d[(d.Sex == sx) & (d.Horizon == h)].sort_values("Age")
            if g.empty:
                _frame(ax, i, j, f"{sx} · {h}   (missing)")
                _recessive(ax)
                continue
            drew = _draw_subject_panel(ax, g, h, sx, arms, floor) or drew
            if k_in_title:
                kt = (int(g["k Controls"].iloc[0]), int(g["k Cases"].iloc[0])) \
                    if "k Controls" in g else (None, int(g.k.iloc[0]))
                tt = (f"{sx} · {h}   ($k_0$={kt[0]}, $k_1$={kt[1]})" if kt[0] is not None
                      else f"{sx} · {h}   ($k$={kt[1]})")
            else:
                tt = f"{sx} · {h}"
            _frame(ax, i, j, tt)
            _recessive(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if drew:
        handles.append(Line2D([], [], color="0.55", lw=1.0, ls=(0, (4, 3))))
        labels.append(f"age {floor:.0f}")
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8.5,
               labelcolor=INK2)
    # With clip_note=False the figure no longer says the interval was truncated, so
    # whatever carries it -- a caption, the surrounding text -- has to.
    fig.suptitle("NHANES: the band, the two curves, and the plug-in at every subject"
                 + ("\nthe interval is truncated at 0.5, the direction of the marker "
                    "being known a priori" if clip_half and clip_note else ""),
                 fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    path = os.path.join(out_dir, "nhanes_bands_subjects"
                        + (f"_k{k}" if k is not None else "")
                        + ("_clipped" if clip_half else "") + ".png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_per_group(out_dir, plugin=True, floor=AGE_FLOOR, rug_dir=None):
    """One band per combination, each at its own neighbourhood sizes.

    The sizes are chosen per combination by select_k_per_group.py and differ between
    panels, so they are named in each title rather than drawn as a dimension: the figure
    is about the six bands, not about k. The reader still needs the sizes, because a
    panel at 15 cases and one at 38 are not equally sharp, and the intervals say so.
    """
    d = pd.read_csv(f"{PRIMARY}/bands_all.csv")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for i, sx in enumerate(SEXES):
        for j, h in enumerate(HORIZONS):
            ax = axes[i, j]
            g = d[(d.Sex == sx) & (d.Horizon == h)].sort_values("Age")
            if g.empty:
                _frame(ax, i, j, f"{sx} · {h}   (no rows)")
                _recessive(ax)
                continue
            pairs = g[["k Controls", "k Cases"]].drop_duplicates()
            if len(pairs) != 1:
                # Selecting one pair per combination is the whole point of this run; if
                # several survived, a silent overlay of two bands is the worst outcome.
                raise ValueError(f"{h}/{sx} holds {len(pairs)} neighbourhood sizes, "
                                 f"expected one:\n{pairs}")
            k0, k1 = int(pairs.iloc[0, 0]), int(pairs.iloc[0, 1])

            drew_rule = _draw_group_panel(ax, g, plugin, floor)
            if rug_dir:
                _cohort_rug(ax, rug_dir, h, sx, float(g.Age.min()),
                            float(g.Age.max()), legend=(i == 0 and j == 0))
            _frame(ax, i, j, f"{sx} · {h}   ({int(g['N Cases'].iloc[0])} cases · "
                             f"$k_0$={k0}, $k_1$={k1})")
            ax.set_ylim(0.12 if rug_dir else 0.2, 1.0)
            _recessive(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if drew_rule:
        handles.append(Line2D([], [], color="0.55", lw=1.0, ls=(0, (4, 3))))
        labels.append(f"age {floor:.0f}")
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
               fontsize=8.5, labelcolor=INK2)
    lo, hi = d.Age.min(), d.Age.max()
    fig.suptitle("NHANES: covariate-specific AUC, neighbourhood sized per combination"
                 + (f", ages {lo:.0f}-{hi:.0f}" if lo > 25 else ""),
                 fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    path = os.path.join(out_dir, "nhanes_bands_per_group"
                        + ("_rug" if rug_dir else "") + ".png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_plugin_fix(out_dir, other_dir):
    """The plug-in before and after exempting the biases from the L2 penalty.

    PRIMARY is the corrected run, `other_dir` the original. The band is drawn once: it is
    rank-based and never touches the networks, so the two runs share it exactly, and any
    daylight between the figures is the plug-in alone.
    """
    a = pd.read_csv(f"{other_dir}/bands_all.csv")
    b = pd.read_csv(f"{PRIMARY}/bands_all.csv")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    inside = {}
    for i, sx in enumerate(SEXES):
        for j, h in enumerate(HORIZONS):
            ax = axes[i, j]
            ga = a[(a.Sex == sx) & (a.Horizon == h)].sort_values("Age")
            gb = b[(b.Sex == sx) & (b.Horizon == h)].sort_values("Age")
            if ga.empty or gb.empty:
                _frame(ax, i, j, f"{sx} · {h}   (missing)")
                _recessive(ax)
                continue
            if not np.allclose(ga.AUC.values, gb.AUC.values):
                raise ValueError(f"{h}/{sx}: the two runs disagree on the band itself, "
                                 f"so they differ in more than the penalty")
            ax.axhline(0.5, color="0.7", lw=0.8, ls=":")
            ax.fill_between(gb.Age, gb["CI95 Lower"], gb["CI95 Upper"], color="0.78",
                            alpha=0.45, lw=0, label="95% interval")
            ax.plot(gb.Age, gb.AUC, color=INK, lw=2.1, label="local k-NN band",
                    path_effects=[pe.Stroke(linewidth=3.6, foreground="white"),
                                  pe.Normal()])
            for g, col, lab in ((ga, HUE["5-year"], "FNN plug-in, penalty on the biases"),
                                (gb, HUE["8-year"], "FNN plug-in, biases exempt")):
                ax.plot(g.Age, g["AUC Parametric"], color=col, lw=1.9, ls=(0, (5, 2)),
                        label=lab)
                n_in = int((((g["AUC Parametric"] >= g["CI95 Lower"])
                             & (g["AUC Parametric"] <= g["CI95 Upper"])).sum()))
                inside.setdefault((sx, h), []).append(n_in)
            was, now = inside[(sx, h)]
            _frame(ax, i, j, f"{sx} · {h}   (inside: {was}/{len(ga)} $\\to$ "
                             f"{now}/{len(gb)})")
            ax.set_ylim(0.3, 0.95)
            _recessive(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9,
               labelcolor=INK2)
    fig.suptitle("NHANES: the FNN plug-in against the band, before and after exempting "
                 "the biases from the L2 penalty", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    path = os.path.join(out_dir, "nhanes_plugin_fix.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _draw_weight_panel(ax, ga, gb, series, floor, shift_label=True):
    """One weighted-vs-unweighted panel. Shared by the grid and the one-file-per-panel
    version so the two cannot drift apart. Returns whether the age rule was drawn.

    floor=None omits the locality cut-off, and shift_label=False the per-panel summary
    of how far weighting moves the estimate -- both for a figure meant to show the two
    curves and nothing else."""
    ax.axhline(0.5, color="0.7", lw=0.8, ls=":")
    drew = _age_rule(ax, gb, floor)
    bits = []
    if series in ("both", "plugin"):
        # The paper's own estimator. Weighting reaches it through the loss, so the two
        # curves come from two differently trained networks.
        for g, ls, lab in ((ga, (0, (5, 2)), "FNN plug-in, unweighted"),
                           (gb, (0, (1.4, 1.4)), "FNN plug-in, weighted")):
            ax.plot(g.Age, g["AUC Parametric"], color=ROLE["plugin"], lw=2.0, ls=ls,
                    label=lab)
        bits.append(f"FNN {(gb['AUC Parametric'] - ga['AUC Parametric']).mean():+.3f}")
    if series in ("both", "knn"):
        for g, col, lab in ((ga, ROLE["knn"], "local k-NN, unweighted"),
                            (gb, ROLE["weighted"], "local k-NN, weighted")):
            ax.plot(g.Age, g.AUC, color=col, lw=2.1, label=lab,
                    path_effects=[pe.Stroke(linewidth=3.6, foreground="white"),
                                  pe.Normal()])
        bits.append(f"k-NN {(gb.AUC - ga.AUC).mean():+.3f}")
    if shift_label:
        ax.annotate("weighted − unweighted\n" + "\n".join(bits),
                    xy=(0.03, 0.03), xycoords="axes fraction", va="bottom", fontsize=8,
                    color=INK2)
    return drew


def figure_weight_auc_split(out_dir, unweighted_dir, k=None, floor=AGE_FLOOR,
                            series="both", suffix="", shift_label=True):
    """The same comparison, one file per horizon/sex instead of a 2x3 grid.

    The y-axis is held common across the six so they stay comparable when read one at a
    time -- which is the whole risk of splitting a grid up, since each panel would
    otherwise be scaled to its own range and a small shift would look like a large one.
    """
    a = pd.read_csv(f"{unweighted_dir}/bands_all.csv")
    b = pd.read_csv(f"{PRIMARY}/bands_all.csv")
    if k is not None:
        a, b = a[a.k == k], b[b.k == k]
    for name, frame in (("unweighted", a), ("weighted", b)):
        n = frame.groupby(["Horizon", "Sex", "Age"]).size()
        if len(n) and n.max() > 1:
            raise ValueError(f"the {name} run has several k at the same age; pass --k")

    drawn = []
    if series in ("both", "plugin"):
        drawn += [a["AUC Parametric"], b["AUC Parametric"]]
    if series in ("both", "knn"):
        drawn += [a.AUC, b.AUC]
    lo_y = min(float(s.min()) for s in drawn)
    hi_y = max(float(s.max()) for s in drawn)
    pad = max(0.03, 0.08 * (hi_y - lo_y))

    paths = []
    for sx in SEXES:
        for h in HORIZONS:
            ga = a[(a.Sex == sx) & (a.Horizon == h)].sort_values("Age")
            gb = b[(b.Sex == sx) & (b.Horizon == h)].sort_values("Age")
            if ga.empty or gb.empty:
                print(f"  {sx} {h}: missing from one of the two runs -- skipped")
                continue
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            drew = _draw_weight_panel(ax, ga, gb, series, floor, shift_label)
            ktag = (f"   ($k$={int(gb.k.iloc[0])})"
                    if series in ("both", "knn") and "k" in gb else "")
            ax.set_xlabel("Age (years)", color=INK2, fontsize=10)
            ax.set_ylabel("AUC", color=INK2, fontsize=10)
            ax.set_ylim(lo_y - pad, hi_y + pad)
            ax.set_title(f"NHANES {sx} · {h}{ktag}\nAUC with and without the survey "
                         f"weights", fontsize=11, color=INK)
            handles, labels = ax.get_legend_handles_labels()
            if drew:
                handles.append(Line2D([], [], color="0.55", lw=1.0, ls=(0, (4, 3))))
                labels.append(f"age {floor:.0f}")
            ax.legend(loc="lower right", frameon=False, fontsize=8,
                      labelcolor=INK2)
            _recessive(ax)
            fig.tight_layout()
            path = os.path.join(
                out_dir,
                f"nhanes_weighted_vs_unweighted_auc{suffix}_{h}_{sx}.png")
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(path)
    return paths


def figure_weight_auc(out_dir, unweighted_dir, k=None, floor=AGE_FLOOR,
                      series="both", suffix="", shift_label=True):
    """The two AUC curves, weighted and not, with no intervals.

    PRIMARY is the weighted run and `unweighted_dir` the one without. Both estimate a
    probability of correct ordering, but not the same one: weighted, the AUC of the
    population the survey represents; unweighted, the AUC of the cohort as sampled. They
    are two estimands, not two estimates of one, so the figure sets them side by side
    rather than treating either as the error of the other.

    The intervals are left out at the caller's request. The curves are close enough in
    places that their difference will read as smaller than it is: at k=30 on the full
    grid the weighted estimate ran 0.055 to 0.092 below the unweighted one in the male
    panels, which is of the order of the half-width.

    Neither curve is the network. Both are the rank-based local estimate, which is why
    the weighted one takes the third colour slot rather than the second: the second is
    the FNN plug-in everywhere else in this set, and reusing it here had readers taking
    the weighted curve for the network.
    """
    a = pd.read_csv(f"{unweighted_dir}/bands_all.csv")
    b = pd.read_csv(f"{PRIMARY}/bands_all.csv")
    # A sweep holds many k against the same ages; one has to be picked or the panel
    # would stack them.
    if k is not None:
        a, b = a[a.k == k], b[b.k == k]
    for name, frame in (("unweighted", a), ("weighted", b)):
        n = frame.groupby(["Horizon", "Sex", "Age"]).size()
        if len(n) and n.max() > 1:
            raise ValueError(f"the {name} run has several k at the same age; pass --k")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for i, sx in enumerate(SEXES):
        for j, h in enumerate(HORIZONS):
            ax = axes[i, j]
            ga = a[(a.Sex == sx) & (a.Horizon == h)].sort_values("Age")
            gb = b[(b.Sex == sx) & (b.Horizon == h)].sort_values("Age")
            if ga.empty or gb.empty:
                _frame(ax, i, j, f"{sx} · {h}   (missing)")
                _recessive(ax)
                continue
            drew = _draw_weight_panel(ax, ga, gb, series, floor, shift_label)
            # k is named only when something on the panel depends on it. The plug-in
            # is fitted once and evaluated on the grid, so it is the same curve at
            # every k; labelling a plug-in-only panel with one would invite the reader
            # to think the curve was computed at that neighbourhood size.
            ktag = (f"   ($k$={int(gb.k.iloc[0])})"
                    if series in ("both", "knn") and "k" in gb else "")
            _frame(ax, i, j, f"{sx} · {h}{ktag}")
            _recessive(ax)
    # Fit the axis to what is actually drawn: the plug-in alone occupies a much narrower
    # band than the pair, and the shared range would leave most of the panel empty.
    drawn = []
    if series in ("both", "plugin"):
        drawn += [a["AUC Parametric"], b["AUC Parametric"]]
    if series in ("both", "knn"):
        drawn += [a.AUC, b.AUC]
    lo_y = min(float(s.min()) for s in drawn)
    hi_y = max(float(s.max()) for s in drawn)
    pad = max(0.03, 0.08 * (hi_y - lo_y))
    for ax in axes.ravel():
        ax.set_ylim(lo_y - pad, hi_y + pad)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if drew:
        handles.append(Line2D([], [], color="0.55", lw=1.0, ls=(0, (4, 3))))
        labels.append(f"age {floor:.0f}")
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               labelcolor=INK2)
    lo, hi = b.Age.min(), b.Age.max()
    fig.suptitle("NHANES: covariate-specific AUC with and without the survey weights"
                 + (f", ages {lo:.0f}-{hi:.0f}" if lo > 25 else "")
                 + "  (point estimates only)", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    path = os.path.join(out_dir, f"nhanes_weighted_vs_unweighted_auc{suffix}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-k", "--k_list", type=int, nargs="+", default=[50],
                   help="neighbourhood sizes to plot (default: 50)")
    p.add_argument("-o", "--out_dir", default="output/knn_bands_real",
                   help="where to write the PNGs")
    p.add_argument("-i", "--in_dir", default=None,
                   help="run directory for the covariate panel (default: "
                        "output/knn_bands_real). The model and residualised runs "
                        "are read from their own directories and are simply omitted "
                        "from the comparison figure when they lack the requested k.")
    p.add_argument("--sweep", action="store_true",
                   help="also draw the three figures that read every k at once: the "
                        "operating-point curves, the age trend against k, and the "
                        "three-k band contrast.")
    p.add_argument("--small_k", type=int, default=10,
                   help="the small neighbourhood in the band contrast (default: 10)")
    p.add_argument("--chosen_k", type=int, default=30,
                   help="the chosen neighbourhood in the band contrast (default: 30)")
    p.add_argument("--weight_auc", default=None, metavar="UNWEIGHTED_DIR",
                   help="-i is the --band_weights run; draw the AUC curves with and "
                        "without the weights, no intervals.")
    p.add_argument("--no_age_rule", action="store_true",
                   help="omit the age cut-off line from the weight figure")
    p.add_argument("--no_shift_label", action="store_true",
                   help="omit the per-panel weighted-minus-unweighted summary")
    p.add_argument("--weight_split", action="store_true",
                   help="write one file per horizon/sex instead of a single 2x3 grid; "
                        "the y-axis stays common across the six.")
    p.add_argument("--weight_suffix", default="",
                   help="appended to the weight figure's filename, so several series "
                        "selections can live side by side")
    p.add_argument("--weight_series", choices=["both", "plugin", "knn"], default="both",
                   help="which estimator the weight comparison shows: the FNN plug-in "
                        "(the paper's estimator), the local k-NN (the band's centre), "
                        "or both.")
    p.add_argument("--plugin_fix", default=None, metavar="ORIGINAL_DIR",
                   help="-i is the --no_bias_decay run; draw it against the original.")
    p.add_argument("--per_group", action="store_true",
                   help="-i holds one neighbourhood size per combination (the output of "
                        "select_k_per_group.py); draw nhanes_bands_per_group.png.")
    p.add_argument("--age_floor", type=float, default=AGE_FLOOR,
                   help="where to draw the locality cut-off; it is omitted when it "
                        "falls outside the reported age range.")
    p.add_argument("--per_group_bmi", action="store_true",
                   help="with --per_group, add a strip under each panel showing the "
                        "cohort in (age, BMI) and the median BMI the grid sits at.")
    p.add_argument("--subject_scatter", nargs="?", type=int, const=-1, default=None,
                   metavar="K",
                   help="draw the band, both curves and one point per subject on one "
                        "set of axes, at this k. Needs the subjects_*.csv that "
                        "local_knn_bands_real.py --subject_auc writes.")
    p.add_argument("--subject_split", action="store_true",
                   help="write the subject-scatter figure as one file per horizon/sex")
    p.add_argument("--no_clip_note", action="store_true",
                   help="omit the subtitle that says the interval was truncated; the "
                        "caption then has to carry it")
    p.add_argument("--no_k_title", action="store_true",
                   help="leave the neighbourhood sizes out of the panel titles in the "
                        "subject-scatter figure")
    p.add_argument("--clip_half", action="store_true",
                   help="truncate the interval at 0.5 in the subject-scatter figure, "
                        "the marker's direction being known a priori.")
    p.add_argument("--subject_arms", action="store_true",
                   help="colour the per-subject points by arm")
    p.add_argument("--per_group_rug", action="store_true",
                   help="with --per_group, draw the subjects as two lanes INSIDE each "
                        "panel instead of in a strip beneath it.")
    p.add_argument("--per_group_under", choices=["bmi", "age"], default="bmi",
                   help="what the strip under each panel shows: the cohort in "
                        "(age, BMI), or the subjects as two jittered lanes along age.")
    p.add_argument("--per_group_split", action="store_true",
                   help="with --per_group, write one file per horizon/sex instead of a "
                        "single 2x3 grid; the y-axis stays common across the six.")
    p.add_argument("--no_plugin", action="store_true",
                   help="leave the FNN plug-in off the per-combination figure.")
    p.add_argument("--k0", type=int, default=None,
                   help="which control-arm size the k-indexed figures use. Only matters "
                        "for an asymmetric run, where several control arms share one "
                        "case-arm k; the default keeps the symmetric rows.")
    p.add_argument("--control_sweep", type=int, default=None, metavar="K1",
                   help="draw the two figures that read the control-arm sweep at this "
                        "fixed case-arm size: nhanes_control_arm_k1_{K1}.png and "
                        "nhanes_k0_contrast_k1_{K1}.png.")
    p.add_argument("-d", "--data_dir", default="./data",
                   help="cohort files, read only to rebuild the control-arm geometry "
                        "that the band CSV does not carry (no model is involved).")
    p.add_argument("--matched", nargs=3, metavar=("UNMATCHED_DIR", "K0", "K1"),
                   default=None,
                   help="-i is the --match_controls run; draws the realized-age map and "
                        "the matched-vs-unmatched comparison at this k0 and k1.")
    p.add_argument("--vs", default=None, metavar="DIR",
                   help="a second run directory to set beside -i, differing from it only "
                        "in whether the survey weights are used; draws "
                        "nhanes_weights_effect_k{K}.png for each -k.")
    args = p.parse_args()

    if args.in_dir:
        set_primary(args.in_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    for k in args.k_list:
        for path in (figure_covariate(k, args.out_dir, args.k0),
                     figure_three(k, args.out_dir, args.k0)):
            if path:
                print("wrote", path)
    if args.weight_auc:
        fn = figure_weight_auc_split if args.weight_split else figure_weight_auc
        out = fn(args.out_dir, args.weight_auc,
                 args.k_list[0] if args.k_list else None,
                 None if args.no_age_rule else args.age_floor,
                 args.weight_series, args.weight_suffix,
                 not args.no_shift_label)
        for path in (out if isinstance(out, list) else [out]):
            if path:
                print("wrote", path)
    if args.plugin_fix:
        path = figure_plugin_fix(args.out_dir, args.plugin_fix)
        if path:
            print("wrote", path)
    if args.subject_scatter:
        fn = (figure_subject_scatter_split if args.subject_split
              else figure_subject_scatter)
        out = fn(args.out_dir,
                 None if args.subject_scatter == -1 else args.subject_scatter,
                 args.age_floor, args.subject_arms, args.clip_half,
                 not args.no_k_title, not args.no_clip_note)
        for path in (out if isinstance(out, list) else [out]):
            if path:
                print("wrote", path)
    if args.per_group:
        if args.per_group_bmi:
            out = figure_per_group_bmi(args.out_dir, args.data_dir,
                                       not args.no_plugin, args.age_floor,
                                       args.per_group_under)
        elif args.per_group_rug:
            out = figure_per_group(args.out_dir, not args.no_plugin, args.age_floor,
                                   args.data_dir)
        else:
            fn = figure_per_group_split if args.per_group_split else figure_per_group
            out = fn(args.out_dir, not args.no_plugin, args.age_floor)
        for path in (out if isinstance(out, list) else [out]):
            if path:
                print("wrote", path)
    if args.control_sweep:
        for path in (figure_control_arm(args.out_dir, args.control_sweep, args.data_dir),
                     figure_k0_contrast(args.out_dir, args.control_sweep)):
            if path:
                print("wrote", path)
    if args.matched:
        other, k0, k1 = args.matched[0], int(args.matched[1]), int(args.matched[2])
        for path in (figure_realized_age(args.out_dir, k0, k1),
                     figure_matched_arms(args.out_dir, other, k0, k1)):
            if path:
                print("wrote", path)
    if args.vs:
        for k in args.k_list:
            path = figure_weight_contrast(args.out_dir, args.vs, k)
            if path:
                print("wrote", path)
    if args.sweep:
        for path in (figure_operating_point(args.out_dir, args.k0),
                     figure_age_slope(args.out_dir, args.k0),
                     figure_k_contrast(args.out_dir, args.small_k, args.chosen_k,
                                       args.k0)):
            if path:
                print("wrote", path)


if __name__ == "__main__":
    main()
