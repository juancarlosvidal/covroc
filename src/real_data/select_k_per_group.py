"""Choose a (control, case) neighbourhood size for each horizon/sex combination.

The six combinations are not one problem. The case arms run from 53 to 234 subjects over
the same sixty-year age range, so a single k gives 21% of the arm in one panel and 9% in
another, and with it a locality that differs by a factor of two between panels that are
read side by side.

The sizes are therefore chosen per combination, by a rule that uses ONLY the covariates:
no outcome, no biomarker, no fitted model enters. That matters for more than tidiness --
a size picked by looking at the resulting bands would make the reported interval an
interval conditional on a selection that its coverage calculation knows nothing about.

    k1 (cases)    the largest whose case-neighbourhood age IQR stays within
                  --iqr_target years, but never below --k1_floor. The floor is not a
                  taste: below about 15 the statistic degenerates -- the k=5 sweep put
                  AUC exactly at 0 in two cells, and the fitted age trend reversed sign
                  between k=5 and k=15 in three combinations.
    k0 (controls) the largest whose control-neighbourhood median age stays within
                  --off_target years of the grid point. Enlarging the control arm costs
                  almost nothing in locality until it does: the control median runs 0.9
                  years from the target at k0=150 and 16 years at the whole arm, and the
                  fitted age trend starts to drift at about k0=500.

Run from the repository root:

    python src/real_data/select_k_per_group.py              # print the table
    python src/real_data/select_k_per_group.py --sh         # emit the six commands
"""
import argparse
import os

import numpy as np
import pandas as pd

COVARIATES = ["RIDAGEYR.x", "BMI"]
AGE_COL = "RIDAGEYR.x"
TARGET_COL = "TAC2"
GENDER_FILES = {"female": "df_f.csv", "male": "df_m.csv"}
HORIZONS = {"tres": "3-year", "cinco": "5-year", "ocho": "8-year"}


def _orders(df, idx, grid, scv):
    """Neighbour orderings of one arm, one per grid point, in scaled covariate space."""
    c = np.column_stack([df[k].to_numpy(float)[idx] for k in COVARIATES]) / scv
    return [np.argsort(np.linalg.norm(c - g / scv, axis=1)) for g in grid]


def choose(data_dir, iqr_target, off_target, k1_floor, grid_points, lo_q, hi_q,
           age_min=None, age_max=None, neighbour_age_min=None):
    rows = []
    for sex, fname in GENDER_FILES.items():
        for horizon, label in HORIZONS.items():
            df = (pd.read_csv(os.path.join(data_dir, fname))
                  .dropna(subset=COVARIATES + [TARGET_COL, horizon])
                  .reset_index(drop=True))
            i0 = np.where(df[horizon].to_numpy() == 0)[0]
            i1 = np.where(df[horizon].to_numpy() == 1)[0]
            # The rule has to see the same pool the estimator will: a floor applied there
            # but not here would size the neighbourhood against subjects it cannot use.
            if neighbour_age_min is not None:
                age_all = df[AGE_COL].to_numpy()
                i0 = i0[age_all[i0] >= neighbour_age_min]
                i1 = i1[age_all[i1] >= neighbour_age_min]
            scv = np.array([df[c].std() for c in COVARIATES], float)
            lo = df[AGE_COL].quantile(lo_q) if age_min is None else float(age_min)
            hi = df[AGE_COL].quantile(hi_q) if age_max is None else float(age_max)
            ages = np.linspace(lo, hi, grid_points)
            grid = np.column_stack([ages, np.full(grid_points,
                                                  df[COVARIATES[1]].median())])
            o0, o1 = _orders(df, i0, grid, scv), _orders(df, i1, grid, scv)
            a0, a1 = df[AGE_COL].to_numpy()[i0], df[AGE_COL].to_numpy()[i1]

            iqr = lambda a, o, k: float(np.mean(
                [np.subtract(*np.percentile(a[oo[:k]], [75, 25])) for oo in o]))
            off = lambda a, o, k: float(np.mean(
                [abs(np.median(a[oo[:k]]) - g) for oo, g in zip(o, ages)]))

            k1 = k1_floor
            for k in range(k1_floor, len(i1) + 1):
                if iqr(a1, o1, k) <= iqr_target:
                    k1 = k
                else:
                    break
            k0 = 20
            for k in range(20, len(i0) + 1, 10):
                if off(a0, o0, k) <= off_target:
                    k0 = k
                else:
                    break
            rows.append({"horizon": horizon, "sex": sex, "label": label,
                         "n_cases": len(i1), "n_controls": len(i0),
                         "k0": k0, "k1": k1,
                         "case_iqr": iqr(a1, o1, k1), "ctrl_iqr": iqr(a0, o0, k0),
                         "ctrl_off": off(a0, o0, k0),
                         "at_floor": k1 == k1_floor and iqr(a1, o1, k1) > iqr_target})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--data_dir", default="./data")
    p.add_argument("--iqr_target", type=float, default=10.0,
                   help="case-neighbourhood age IQR to stay within, years (default: 10)")
    p.add_argument("--off_target", type=float, default=1.0,
                   help="how far the control median age may sit from the grid point, "
                        "years (default: 1)")
    p.add_argument("--k1_floor", type=int, default=15,
                   help="never go below this many cases (default: 15)")
    p.add_argument("-gp", "--grid_points", type=int, default=21)
    p.add_argument("--lo_q", type=float, default=0.025)
    p.add_argument("--hi_q", type=float, default=0.975)
    p.add_argument("--age_min", type=float, default=None,
                   help="first grid age, overriding --lo_q")
    p.add_argument("--age_max", type=float, default=None,
                   help="last grid age, overriding --hi_q")
    p.add_argument("--neighbour_age_min", type=float, default=None,
                   help="age floor on the pool, matching the estimator's own")
    p.add_argument("--sh", action="store_true",
                   help="emit the six local_knn_bands_real.py commands instead")
    a = p.parse_args()

    t = choose(a.data_dir, a.iqr_target, a.off_target, a.k1_floor,
               a.grid_points, a.lo_q, a.hi_q, a.age_min, a.age_max,
               a.neighbour_age_min)
    if a.sh:
        for r in t.itertuples():
            print(f"-g {r.horizon} -s {r.sex} -k {r.k0}:{r.k1}")
        return
    print(f"k1: largest with case age IQR <= {a.iqr_target:g} yr, floor {a.k1_floor}")
    print(f"k0: largest with control median age within {a.off_target:g} yr of target\n")
    print(f"{'combo':16s}{'cases':>7}{'k1':>5}{'%arm':>7}{'caseIQR':>9}"
          f"{'k0':>6}{'%arm':>7}{'ctrlOff':>9}{'ctrlIQR':>9}")
    for r in t.itertuples():
        flag = " *" if r.at_floor else ""
        print(f"{r.label + ' ' + r.sex:16s}{r.n_cases:>7}{r.k1:>5}"
              f"{100 * r.k1 / r.n_cases:>6.0f}%{r.case_iqr:>9.1f}{r.k0:>6}"
              f"{100 * r.k0 / r.n_controls:>6.0f}%{r.ctrl_off:>9.2f}"
              f"{r.ctrl_iqr:>9.1f}{flag}")
    if t.at_floor.any():
        print(f"\n* held at the k1 floor of {a.k1_floor}: the IQR target of "
              f"{a.iqr_target:g} yr would have forced a smaller, degenerate arm.")


if __name__ == "__main__":
    main()
