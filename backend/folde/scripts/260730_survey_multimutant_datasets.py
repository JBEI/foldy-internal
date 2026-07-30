"""Survey ProteinGym substitution DMS files for long-range multi-mutant content.

Motivation: SPG1_STRSG_Wu_2016 is a 4-site combinatorial library (V39/D40/G41/V54
on GB1). It has 121k quadruple mutants, but they occupy only four mutually
contacting positions, so it measures local epistasis in one tiny neighborhood.
It cannot tell you whether a model generalizes across long-range multi-mutants.

The discriminating statistic is therefore NOT the mutation-order histogram --
it is how many distinct positions the library touches and how far apart the
co-mutated positions are in sequence. This script reports, per dataset:

  n_multi      variants with >= 2 substitutions
  frac_multi   fraction of the dataset that is multi-mutant
  n_pos        distinct positions mutated anywhere in the dataset
  n_pos_multi  distinct positions involved in multi-mutants
  med_span     median (max_pos - min_pos) within a multi-mutant
  p90_span     90th-percentile span
  combinatorial  n_pos_multi <= 6, i.e. a fixed-site library like Wu 2016

Sequence separation is a proxy for structural contact, not a substitute. A large
span is strong evidence positions are NOT mutually contacting; a small span does
not prove they are. Treat this as a screen, then check structures on the finalists.

Run:
  cd backend && ../.venv/bin/python -u folde/scripts/260730_survey_multimutant_datasets.py
"""

import os
import re
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("FOLDE_CONSTANT_LIAR_DEVICE", "cpu")

from folde.data import DMS_DIR  # noqa: E402

POS_RE = re.compile(r"^[A-Z](\d+)[A-Z]$")

# Below this many distinct multi-mutant positions, the library is a fixed-site
# combinatorial scan (Wu 2016 has exactly 4) rather than a distributed library.
COMBINATORIAL_POS_CUTOFF = 6


def positions(mutant: str):
    out = []
    for allele in str(mutant).split(":"):
        m = POS_RE.match(allele.strip())
        if m:
            out.append(int(m.group(1)))
    return out


def survey_one(path):
    df = pd.read_csv(path, usecols=["mutant"])
    mutants = df["mutant"].astype(str)
    orders = mutants.str.count(":") + 1
    n_multi = int((orders >= 2).sum())
    if n_multi == 0:
        return None

    all_pos = set()
    multi_pos = set()
    spans = []
    for mutant in mutants[orders >= 2]:
        p = positions(mutant)
        if len(p) < 2:
            continue
        multi_pos.update(p)
        spans.append(max(p) - min(p))
    for mutant in mutants[orders == 1]:
        all_pos.update(positions(mutant))
    all_pos |= multi_pos

    spans_arr = np.asarray(spans) if spans else np.zeros(1)
    return {
        "dataset": os.path.basename(path)[:-4],
        "n": len(df),
        "n_multi": n_multi,
        "frac_multi": n_multi / len(df),
        "max_order": int(orders.max()),
        "n_pos": len(all_pos),
        "n_pos_multi": len(multi_pos),
        "med_span": float(np.median(spans_arr)),
        "p90_span": float(np.percentile(spans_arr, 90)),
    }


def main() -> int:
    rows = []
    files = sorted(f for f in os.listdir(DMS_DIR) if f.endswith(".csv"))
    for i, fname in enumerate(files, 1):
        try:
            row = survey_one(os.path.join(DMS_DIR, fname))
        except Exception as exc:  # noqa: BLE001 - a malformed file must not abort the survey
            print(f"  ! {fname}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if row:
            rows.append(row)
        if i % 50 == 0:
            print(f"  ...{i}/{len(files)}", file=sys.stderr)

    out = pd.DataFrame(rows).sort_values("n_multi", ascending=False)
    out["combinatorial"] = out["n_pos_multi"] <= COMBINATORIAL_POS_CUTOFF

    pd.set_option("display.width", 200)
    print(f"\n{len(out)} of {len(files)} datasets contain multi-mutants\n")
    cols = [
        "dataset",
        "n",
        "n_multi",
        "frac_multi",
        "max_order",
        "n_pos_multi",
        "med_span",
        "p90_span",
        "combinatorial",
    ]
    print(out[cols].head(40).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    dist = out[(~out["combinatorial"]) & (out["n_multi"] >= 1000)]
    print(
        f"\n\n=== DISTRIBUTED multi-mutant libraries (n_pos_multi > "
        f"{COMBINATORIAL_POS_CUTOFF}, n_multi >= 1000) ===\n"
    )
    print(dist[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    dest = "folde/model_evals/260730-multimutant-dataset-survey.csv"
    out.to_csv(dest, index=False)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
