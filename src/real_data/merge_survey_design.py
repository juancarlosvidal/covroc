"""Merge the NHANES survey-design variables into the working case-study files.

The analysis files data/df_{m,f}.csv carry the marker and the outcome but not the
survey design. data/data_0.csv carries the design, the full covariate set and the
accelerometry profile, but neither the outcome nor the marker. This script joins
them and writes data/df_{m,f}_full.csv, which is df_{m,f}.csv plus:

    wtmec4yr_adj_norm   4-year MEC weight, normalised to mean 1, written ALSO as
                        a column literally named "weights" because that is the
                        name src/real_data/local_knn_bands_real.py looks for
    wtmec4yr_adj        the same weight unnormalised
    WTMEC2YR            the official 2-year MEC weight it derives from
    SDMVPSU, SDMVSTRA   masked variance units and strata
    SDDSRVYR            cycle (7 = 2011-12, 8 = 2013-14)
    SEQN                respondent sequence number, so the join can be redone

WHICH WEIGHT. The cohort pools two cycles, so the 2-year MEC weight must be halved:
wtmec4yr_adj is exactly WTMEC2YR/2, and wtmec4yr_adj_norm rescales it to mean 1,
which is the form to pass as sample weights to a loss. Do NOT use WTINT2YR: its
maximum is 33.6 million against a mean of 71,810, i.e. 468x, while WTMEC2YR stays
inside 5,650-222,580, so the interview weight has at least one corrupt value.
Do NOT use WT either -- despite the name it sits in the accelerometry block
(TAC, TLAC, WT, ST, MVPA) and correlates 0.55 with TAC; it is wear time.

THE JOIN. data_0.csv has SEQN but df_{m,f}.csv do not, so the key is built from
(RIAGENDR, RIDAGEYR.x, TAC) rounded to 3 decimals. That key is unique in both files
and matches all 5,006 working subjects exactly, with 5 subjects left over in
data_0 -- all with TAC between 5,000 and 12,800 against a cohort mean of 1.45
million, i.e. essentially no recorded activity, and presumably dropped for
insufficient wear time. They are reported, not merged, since the working files
define the analysis sample.

NOT MERGED. data_0's own TAC column is not the marker: the manuscript's TAC is
df_{m,f}'s TAC2, which no column of data_0 reproduces (the closest, TLAC, reaches
a Spearman correlation of 0.68). TAC and TAC2 correlate only 0.51 in rank, so they
are different quantities and the AUC, which depends on ranks alone, differs between
them. This script therefore leaves TAC2 untouched.

Run from the repository root:

    python src/real_data/merge_survey_design.py
"""
import argparse
import os

import numpy as np
import pandas as pd

DESIGN = ["SEQN", "WTMEC2YR", "wtmec4yr_adj", "wtmec4yr_adj_norm",
          "SDMVPSU", "SDMVSTRA", "SDDSRVYR"]
WEIGHT = "wtmec4yr_adj_norm"
GENDER_FILES = {"female": ("df_f.csv", "df_f_full.csv"),
                "male": ("df_m.csv", "df_m_full.csv")}


def key_of(df):
    return (df["RIAGENDR"].astype(int).astype(str) + "_"
            + df["RIDAGEYR.x"].astype(float).round(3).astype(str) + "_"
            + df["TAC"].astype(float).round(3).astype(str))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input_dir", default="./data")
    p.add_argument("--full", default="data_0.csv",
                   help="the file carrying the survey design")
    args = p.parse_args()

    full = pd.read_csv(os.path.join(args.input_dir, args.full), low_memory=False)
    missing = [c for c in DESIGN if c not in full.columns]
    if missing:
        raise SystemExit(f"{args.full} lacks {missing}")
    full = full.copy()
    full["_key"] = key_of(full)
    if full["_key"].duplicated().any():
        raise SystemExit("the join key is not unique in the design file")

    design = full[["_key"] + DESIGN].copy()
    seen = set()
    for sex, (src, dest) in GENDER_FILES.items():
        df = pd.read_csv(os.path.join(args.input_dir, src))
        df["_key"] = key_of(df)
        if df["_key"].duplicated().any():
            raise SystemExit(f"the join key is not unique in {src}")

        out = df.merge(design, on="_key", how="left", validate="one_to_one")
        unmatched = out[WEIGHT].isna().sum()
        if unmatched:
            raise SystemExit(f"{src}: {unmatched} of {len(df)} subjects did not match")
        seen.update(out["_key"])

        # The name local_knn_bands_real.py's build_inputs looks for.
        out["weights"] = out[WEIGHT]
        out = out.drop(columns=["_key"])
        path = os.path.join(args.input_dir, dest)
        out.to_csv(path, index=False)
        w = out["weights"]
        print(f"{sex:<7} {len(out)} subjects -> {path}")
        print(f"        weights: mean {w.mean():.3f}  sd {w.std():.3f}  "
              f"range {w.min():.2f}-{w.max():.2f}  "
              f"ratio max/min {w.max()/w.min():.1f}")
        print(f"        design : {out.SDMVSTRA.nunique()} strata, "
              f"{out.groupby('SDMVSTRA').SDMVPSU.nunique().max()} PSUs per stratum max, "
              f"cycles {sorted(out.SDDSRVYR.unique())}")

    left = full[~full["_key"].isin(seen)]
    print(f"\n{len(left)} subject(s) in {args.full} are not in the working files:")
    if len(left):
        print(left[["SEQN", "RIAGENDR", "RIDAGEYR.x", "BMXBMI", "TAC"]]
              .to_string(index=False))
        print("  (TAC far below the cohort mean of ~1.45e6 -- presumably excluded "
              "for insufficient wear time)")


if __name__ == "__main__":
    main()
