"""Flag duplicate and unchanged data without removing it.

    uv run python scripts/flag_duplicates.py

Policy (team decision, 2026-10-04): no rows are removed. Repeated content across files is the
phenomenon RQ1 studies, and file-level analyses already use one version per file. Data-level
duplicates are flagged so that analyses can report or exclude them explicitly:

- whitespace-only versions: content identical to the previous version after normalization
- reverts: content identical to an earlier, non-consecutive version of the same file
- forks: the same repository name under two owners, sharing at least one non-stub body
  (same-name repositories without shared content are name collisions, not forks)
- possible renames: in one repository, a file whose first body is near-identical to another
  file's last body, starting after that file's last observed version (GHAW-H keeps one path
  per history and does not record deletions, so a rename shows up as two files)

Writes to results/quality/:
1. version_flags.csv: one row per version with `whitespace_only` and `revert_of_rank`.
2. repo_flags.csv: repositories sharing a name across owners, with `fork` true or false.
3. rename_candidates.csv: possible renames within a repository.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ghaw_relations.data import load_snapshots
from ghaw_relations.relations import fit_tfidf, is_stub, text_hash

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "quality"
H = "source_markdown_file_history_id"
RENAME_NEAR = 0.9  # placeholder: body cosine for a possible rename (same as near-identical)


def version_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([H, "rank"])
    prev = df.groupby(H).content_hash.shift()
    first_rank = df.groupby([H, "content_hash"])["rank"].transform("min")
    out = df[["source_markdown_file_version_id", H, "repo_full_name", "path", "rank", "committed_at"]].copy()
    out["whitespace_only"] = df.content_hash.eq(prev).to_numpy()
    revert = (first_rank < df["rank"]) & df.content_hash.ne(prev)
    out["revert_of_rank"] = np.where(revert, first_rank, np.nan)
    return out.rename(columns={"source_markdown_file_version_id": "version_id", H: "history_id", "repo_full_name": "repo"})


def repo_flags(df: pd.DataFrame) -> pd.DataFrame:
    name = df.repo_full_name.str.split("/").str[1].str.lower()
    ns = df[~df.body.map(is_stub)]
    bodies = ns.groupby("repo_full_name").body_hash.agg(set)
    rows = []
    for n, repos in df.groupby(name).repo_full_name.unique().items():
        if len(repos) < 2:
            continue
        repos = sorted(repos)
        shared = set.intersection(*(bodies.get(r, set()) for r in repos))
        rows += [{"name": n, "repo": r, "same_name_repos": "; ".join(x for x in repos if x != r),
                  "shared_nonstub_bodies": len(shared), "fork": bool(shared)} for r in repos]
    return pd.DataFrame(rows)


def rename_candidates(df: pd.DataFrame) -> pd.DataFrame:
    ns = df[~df.body.map(is_stub)].sort_values([H, "rank"])
    first, last = ns.groupby(H).head(1).reset_index(drop=True), ns.groupby(H).tail(1).reset_index(drop=True)
    vec = fit_tfidf(df.body)
    sim = (vec.transform(first.body) @ vec.transform(last.body).T).toarray()
    end = df.groupby(H).committed_at.max()
    start = df.groupby(H).committed_at.min()
    rows = []
    for i, b in first.iterrows():
        for j in np.where(last.repo_full_name.eq(b.repo_full_name).to_numpy() & (sim[i] >= RENAME_NEAR))[0]:
            a = last.iloc[j]
            if a[H] == b[H] or end[a[H]] > b.committed_at:
                continue
            if start[a[H]] == start[b[H]]:  # created together: parallel variants, not a rename
                continue
            rows.append({"repo": b.repo_full_name, "ended_history_id": a[H], "ended_path": a.path,
                         "started_history_id": b[H], "started_path": b.path, "body_cos": round(float(sim[i, j]), 4),
                         "gap_days": round((b.committed_at - end[a[H]]).total_seconds() / 86400, 1)})
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_snapshots()
    df["committed_at"] = pd.to_datetime(df.committed_at, utc=True)
    df["content_hash"] = df.content.map(text_hash)
    df["body_hash"] = df.body.map(text_hash)

    vf, rf, rc = version_flags(df), repo_flags(df), rename_candidates(df)
    vf.to_csv(OUT_DIR / "version_flags.csv", index=False)
    rf.to_csv(OUT_DIR / "repo_flags.csv", index=False)
    rc.to_csv(OUT_DIR / "rename_candidates.csv", index=False)

    print(f"versions: {len(vf)} | whitespace-only: {vf.whitespace_only.sum()} | "
          f"reverts: {vf.revert_of_rank.notna().sum()} in {vf[vf.revert_of_rank.notna()].history_id.nunique()} files")
    print("repos sharing a name across owners:")
    print(rf[["repo", "same_name_repos", "shared_nonstub_bodies", "fork"]].to_string(index=False))
    print("possible renames:")
    print(rc[["repo", "ended_path", "started_path", "body_cos", "gap_days"]].to_string(index=False) if len(rc) else "none")


if __name__ == "__main__":
    main()
