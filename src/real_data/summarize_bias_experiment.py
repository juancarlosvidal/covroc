"""Did exempting the biases from the L2 penalty help, or was it seed noise?

Reads output/bias_decay_experiment/<variant>/seed<N>/bands_<horizon>_<sex>.csv, written by
hpc/py_bias_decay_array.sh, and reports per combination:

    excess      mean over the age grid of (FNN plug-in AUC - local band AUC). This is the
                quantity in question: the plug-in sits above the rank-based band and we
                want to know whether the fix brings it down.
    d-mu1       mean (fitted case mean - local case mean), in units of the local control
                SD. The mechanism: the control arm is recovered to within 0.09 SD, the
                case arm is pulled 0.35-0.66 SD low, and the whole excess follows from it.

Each is averaged over seeds, with the standard error OF THAT MEAN, so the comparison
carries its own precision. The laptop run used one seed per cell and could not: its swings
(-0.081 to +0.031) were the size of the 0.044 that a change of seed alone moves this
estimate.

Run from the repository root:

    python src/real_data/summarize_bias_experiment.py
    python src/real_data/summarize_bias_experiment.py -i some/other/root --csv out.csv
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


def load(root):
    rows = []
    pattern = os.path.join(root, "*", "seed*", "bands_*_*.csv")
    for path in sorted(glob.glob(pattern)):
        m = re.search(r"/([^/]+)/seed(\d+)/bands_([a-z]+)_([a-z]+)\.csv$", path)
        if not m:
            continue
        variant, seed, _, _ = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        d = pd.read_csv(path)
        if d.empty:
            continue
        s0 = d["Sigma0 Local"].mean()
        rows.append({
            "variant": variant, "seed": seed,
            "combo": f"{d.Horizon.iloc[0]} {d.Sex.iloc[0]}",
            "excess": float((d["AUC Parametric"] - d["AUC"]).mean()),
            "d_mu1": float((d["Mu1 Net"] - d["Mu1 Local"]).mean() / s0),
            "d_mu0": float((d["Mu0 Net"] - d["Mu0 Local"]).mean() / s0),
            "band": float(d["AUC"].mean()),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--root", default="output/bias_decay_experiment")
    p.add_argument("--csv", default=None, help="also write the per-run table here")
    p.add_argument("--reference", default=None,
                   help="variant every other is compared against; the first "
                        "alphabetically by default")
    a = p.parse_args()

    d = load(a.root)
    if d.empty:
        raise SystemExit(f"no runs found under {a.root}")
    if a.csv:
        d.to_csv(a.csv, index=False)
        print(f"per-run table -> {a.csv}\n")

    seeds = sorted(d.seed.unique())
    print(f"{len(d)} runs | variants {sorted(d.variant.unique())} | seeds {seeds}\n")

    # The band never touches the networks, so it must not move with seed or variant.
    # If it does, something other than the penalty changed and the comparison is void.
    spread = d.groupby("combo").band.agg(lambda x: x.max() - x.min())
    if (spread > 1e-9).any():
        print("!! the rank-based band differs across runs, which it cannot do if only")
        print("   the penalty changed. Largest spread by combination:")
        print(spread[spread > 1e-9].round(6).to_string(), "\n")

    # The instability of a single fit, which is what decides whether a per-combination
    # excess means anything. Reported before the variant comparison because it does not
    # need both variants, and because it sets the resolution of everything below.
    print("spread of ONE fit across seeds (sigma), per variant")
    print(f"{'combo':16s}{'variant':14s}{'seeds':>6}{'mean excess':>13}"
          f"{'sigma':>8}{'sigma of plug-in level':>24}")
    for (combo, variant), g in d.groupby(["combo", "variant"]):
        n = len(g)
        sd = g.excess.std(ddof=1) if n > 1 else float("nan")
        print(f"{combo:16s}{variant:14s}{n:>6}{g.excess.mean():>+13.3f}{sd:>8.3f}"
              f"{(g.d_mu1.std(ddof=1) if n > 1 else float('nan')):>24.3f}")
    pooled = d.groupby(["combo", "variant"]).excess.std(ddof=1).mean()
    print(f"\npooled sigma of a single fit: {pooled:.3f}")
    print(f"mean excess over all cells:   {d.excess.mean():+.3f}  "
          f"({abs(d.excess.mean()) / pooled:.1f} sigma)\n")

    # Each variant against a reference. With two variants this is the original
    # comparison; with more -- a sweep over one hyperparameter -- it is each setting
    # against the one it has to beat.
    variants = sorted(d.variant.unique())
    ref = a.reference if a.reference else variants[0]
    if ref not in variants:
        raise SystemExit(f"reference {ref!r} not among {variants}")
    others = [v for v in variants if v != ref]
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan

    print(f"each variant against {ref}, mean over seeds +- the standard error of "
          f"that mean")
    print(f"{'combo':16s}{ref:>20s}" + "".join(f"{v:>22s}" for v in others))
    pooled = {v: [] for v in others}
    for combo, g in d.groupby("combo"):
        cells = {v: gg.excess.values for v, gg in g.groupby("variant")}
        if ref not in cells:
            print(f"{combo:16s}  no {ref} -- skipped")
            continue
        r = cells[ref]
        line = f"{combo:16s}{r.mean():>+13.3f} ±{se(r):<5.3f}"
        for v in others:
            if v not in cells:
                line += f"{'--':>22s}"
                continue
            dif = cells[v].mean() - r.mean()
            dse = np.sqrt(se(r) ** 2 + se(cells[v]) ** 2)
            pooled[v].append(dif)
            star = "*" if np.isfinite(dse) and abs(dif) >= 2 * dse else " "
            line += f"{dif:>+15.3f} ±{dse:<4.3f}{star}"
        print(line)
    print("\n* the difference exceeds twice its own standard error")
    for v in others:
        if pooled[v]:
            print(f"pooled over combinations, {v} - {ref}: {np.mean(pooled[v]):+.3f}")

    # The sweep's own criterion: which setting gives the least variable fit. Reported
    # apart from the excess on purpose -- it is an optimisation diagnostic, and choosing
    # by it does not select on the quantity under study.
    print("\nspread of a single fit, pooled over combinations (lower is steadier):")
    for v, s in d.groupby("variant").apply(
            lambda g: g.groupby("combo").excess.std(ddof=1).mean()).sort_values().items():
        print(f"  {v:16s}{s:.4f}" + ("   <- least variable" if s == min(
            d.groupby("variant").apply(
                lambda g: g.groupby("combo").excess.std(ddof=1).mean())) else ""))

    print("\nfitted case mean, in local control SDs (the mechanism):")
    piv = d.pivot_table(index="combo", columns="variant", values="d_mu1",
                        aggfunc=["mean", "std"])
    print(piv.round(3).to_string())
    print("\nfitted control mean, same units (should be near zero throughout):")
    print(d.pivot_table(index="combo", columns="variant", values="d_mu0",
                        aggfunc="mean").round(3).to_string())


if __name__ == "__main__":
    main()
