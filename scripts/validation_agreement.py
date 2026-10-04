"""Agreement between manual labels and the automatic classification (validation rounds).

    uv run python scripts/validation_agreement.py

Round 1 (2026-10-04): one coder labeled a stratified sample of 87 detected relations (seed 261004)
in blind mode, in a local comparison viewer that is not part of this repository. Each label records
the coder's *first* verdict, given before the machine verdict was shown; statistics use it.
Case C073 is excluded: the viewer displayed an emptied template revision for it (a display error,
since fixed), so its verdict does not judge the relation.

Inputs: validation/round1/sample_cases.csv (strata, machine verdicts) and labels_coder1.csv.
Writes results/validation/round1_agreement.csv (overall) and round1_by_stratum.csv.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROUND1 = ROOT / "validation" / "round1"
OUT = ROOT / "results" / "validation"
EXCLUDED = {"C073": "display error: the viewer showed an emptied template revision"}


def kappa(human: pd.Series, machine: pd.Series) -> float:
    po = (human == machine).mean()
    ph, pm = (human == "related").mean(), (machine == "related").mean()
    pe = ph * pm + (1 - ph) * (1 - pm)
    return (po - pe) / (1 - pe)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(ROUND1 / "sample_cases.csv")
    labels = pd.read_csv(ROUND1 / "labels_coder1.csv")
    d = cases[["case_id", "stratum", "machine_verdict"]].merge(
        labels[["case_id", "first_verdict", "verdict", "blind_at_label"]], on="case_id", how="left")
    d["excluded"] = d.case_id.map(EXCLUDED)
    used = d[d.excluded.isna() & d.first_verdict.notna()]
    comp = used[used.first_verdict != "unsure"]
    agree = comp.first_verdict == comp.machine_verdict
    overall = pd.DataFrame([{
        "round": 1, "coders": 1, "cases": len(d), "labeled": int(d.first_verdict.notna().sum()),
        "excluded": len(d) - len(used), "unsure": int((used.first_verdict == "unsure").sum()),
        "compared": len(comp), "agree": int(agree.sum()), "agreement": round(agree.mean(), 4),
        "cohen_kappa_vs_machine": round(kappa(comp.first_verdict, comp.machine_verdict), 4),
        "all_blind": bool(used.blind_at_label.all()),
        "changed_after_reveal": int((used.first_verdict != used.verdict).sum()),
    }])
    u = used.assign(agree=used.first_verdict == used.machine_verdict, unsure=used.first_verdict == "unsure")
    u["disagree"] = ~u.agree & ~u.unsure
    by = u.groupby(["stratum", "machine_verdict"]).agg(cases=("case_id", "size"), agree=("agree", "sum"),
                                                        disagree=("disagree", "sum"), unsure=("unsure", "sum")).reset_index()
    overall.to_csv(OUT / "round1_agreement.csv", index=False)
    by.to_csv(OUT / "round1_by_stratum.csv", index=False)
    print(overall.T.to_string(header=False))
    print(pd.crosstab(comp.first_verdict, comp.machine_verdict).to_string())
    print(by.to_string(index=False))


if __name__ == "__main__":
    main()
