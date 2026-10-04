"""Version-level evolution of GH-AW instruction files (RQ1, Stage 2).

Uses every version, not only the latest one, and writes CSVs to results/evolution/:

1. version_evolution.csv: each version against its own history (previous and first version),
   with frontmatter keys added or removed and changes to the declared `source:`.
2. history_evolution.csv: one row per file history summarizing how it evolved.
3. prior_matches.csv: for each version, the most similar version committed *earlier* in another
   history, split by scope (same repo / same owner / other owner). This is a candidate earlier
   relative, not necessarily the parent: templates outside the dataset are not covered.
   Stub bodies (see ghaw_relations.relations.is_stub) are excluded on both sides.
4. origin_tracking.csv: for files whose first version closely matches an earlier file, whether
   later versions follow that file's updates, stay close to it, or drift away.

Usage:
    uv run python scripts/trace_evolution.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ghaw_relations.data import load_snapshots
from ghaw_relations.relations import fit_tfidf, frontmatter_keys, frontmatter_source, is_stub, text_hash

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "evolution"
ORIGIN_THRESHOLD = 0.5  # placeholder: first-version prior match needed to track a file against a relative
DRIFT_DELTA = 0.1  # placeholder: drop in cosine to the origin version that counts as drifting away
SCOPES = ("same_repo", "same_owner", "other_owner")


def _str_or_none(value) -> str | None:
    """Missing `source:` values come back as None or NaN; treat both as absent."""
    return value if isinstance(value, str) else None


def prepare() -> pd.DataFrame:
    df = load_snapshots()
    df["owner"] = df.repo_full_name.str.split("/").str[0]
    df["filename"] = df.path.str.rsplit("/", n=1).str[-1]
    df["committed_at"] = pd.to_datetime(df.committed_at, utc=True)
    df["fm_hash"] = df.frontmatter.map(text_hash)
    df["body_hash"] = df.body.map(text_hash)
    df["fm_keys"] = df.frontmatter.map(frontmatter_keys)
    df["source"] = df.frontmatter.map(frontmatter_source)
    df["stub"] = df.body.map(is_stub)
    return df


def within_history(df: pd.DataFrame, sim: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    hist = df.source_markdown_file_history_id
    first_pos = df.groupby(hist).cumcount().eq(0)
    first_idx = pd.Series(np.where(first_pos)[0], index=hist[first_pos]).reindex(hist).to_numpy()
    prev_idx = np.r_[-1, np.arange(len(df) - 1)]
    has_prev = hist.eq(hist.shift()).to_numpy()

    rows = []
    for i, row in enumerate(df.itertuples()):
        out = {
            "version_id": row.source_markdown_file_version_id,
            "history_id": row.source_markdown_file_history_id,
            "repo": row.repo_full_name,
            "filename": row.filename,
            "rank": row.rank,
            "committed_at": row.committed_at,
            "body_cos_first": round(float(sim[i, first_idx[i]]), 4),
            "stub": row.stub,
            "declared_source": row.source,
        }
        if has_prev[i]:
            p = df.iloc[prev_idx[i]]
            fm_same, body_same = p.fm_hash == row.fm_hash, p.body_hash == row.body_hash
            prev_keys, keys = p.fm_keys or frozenset(), row.fm_keys or frozenset()
            out |= {
                "change": {(True, True): "no change", (False, True): "frontmatter only",
                           (True, False): "body only", (False, False): "both"}[(fm_same, body_same)],
                "body_cos_prev": round(float(sim[i, prev_idx[i]]), 4),
                "days_since_prev": round((row.committed_at - p.committed_at).total_seconds() / 86400, 2),
                "fm_keys_added": "; ".join(sorted(keys - prev_keys)),
                "fm_keys_removed": "; ".join(sorted(prev_keys - keys)),
                "source_changed": _str_or_none(p.source) != _str_or_none(row.source),
            }
        rows.append(out)
    versions = pd.DataFrame(rows)

    g = versions.groupby("history_id")
    histories = g.agg(
        repo=("repo", "first"),
        filename=("filename", "first"),
        n_versions=("rank", "size"),
        first_commit=("committed_at", "min"),
        last_commit=("committed_at", "max"),
        n_frontmatter_only=("change", lambda s: (s == "frontmatter only").sum()),
        n_body_changes=("change", lambda s: s.isin(["body only", "both"]).sum()),
        body_cos_first_last=("body_cos_first", "last"),
        min_body_cos_first=("body_cos_first", "min"),
        declared_source_first=("declared_source", "first"),
        declared_source_last=("declared_source", "last"),
        n_source_changes=("source_changed", lambda s: s.fillna(False).astype(bool).sum()),
        n_stub_versions=("stub", "sum"),
    ).reset_index()
    histories["span_days"] = ((histories.last_commit - histories.first_commit).dt.total_seconds() / 86400).round(1)
    return versions, histories


def prior_matches(df: pd.DataFrame, sim: np.ndarray) -> pd.DataFrame:
    """Most similar version in another history, committed strictly earlier, per scope."""
    hist = df.source_markdown_file_history_id.to_numpy()
    repo = df.repository_id.to_numpy()
    owner = df.owner.to_numpy()
    t = df.committed_at.dt.tz_convert(None).to_numpy()
    stub = df.stub.to_numpy()
    earlier = (t[None, :] < t[:, None]) & (hist[None, :] != hist[:, None]) & ~stub[None, :] & ~stub[:, None]
    same_repo = repo[None, :] == repo[:, None]
    same_owner = owner[None, :] == owner[:, None]
    masks = {
        "same_repo": earlier & same_repo,
        "same_owner": earlier & same_owner & ~same_repo,
        "other_owner": earlier & ~same_owner,
    }
    out = df[["source_markdown_file_version_id", "source_markdown_file_history_id", "repo_full_name", "filename", "rank"]].copy()
    out.columns = ["version_id", "history_id", "repo", "filename", "rank"]
    for scope, mask in masks.items():
        s = np.where(mask, sim, -1.0)
        j = s.argmax(axis=1)
        found = s[np.arange(len(df)), j] >= 0
        out[f"{scope}_cos"] = np.where(found, s[np.arange(len(df)), j].round(4), np.nan)
        out[f"{scope}_version_id"] = np.where(found, df.source_markdown_file_version_id.to_numpy()[j], np.nan)
        out[f"{scope}_repo"] = np.where(found, df.repo_full_name.to_numpy()[j], None)
        out[f"{scope}_filename"] = np.where(found, df.filename.to_numpy()[j], None)
        out[f"{scope}_lag_days"] = np.where(found, ((t - t[j]) / np.timedelta64(1, "D")).round(2), np.nan)
    best = out[[f"{s}_cos" for s in SCOPES]].fillna(-1)
    out["best_scope"] = np.where(best.max(axis=1) >= 0, best.idxmax(axis=1).str.removesuffix("_cos"), None)
    out["best_cos"] = best.max(axis=1).where(lambda x: x >= 0)
    return out


def origin_tracking(df: pd.DataFrame, sim: np.ndarray, prior: pd.DataFrame) -> pd.DataFrame:
    """For files whose first version closely matches an earlier version elsewhere, follow the pair over time."""
    pos = {v: i for i, v in enumerate(df.source_markdown_file_version_id)}
    t = df.committed_at.dt.tz_convert(None).to_numpy()
    firsts = prior.groupby("history_id").head(1)
    rows = []
    for row in firsts.itertuples():
        if not row.best_cos >= ORIGIN_THRESHOLD:
            continue
        origin_vid = int(getattr(row, f"{row.best_scope}_version_id"))
        origin_hist = df.source_markdown_file_history_id.iloc[pos[origin_vid]]
        parent = np.where(df.source_markdown_file_history_id.to_numpy() == origin_hist)[0]
        child = np.where(df.source_markdown_file_history_id.to_numpy() == row.history_id)[0]
        last = child[-1]
        # relative's version that was current when the child's last version was committed
        parent_then = parent[t[parent] <= t[last]][-1]
        # which of the relative's versions (up to then) the child's first / last version resembles most
        avail_first = parent[t[parent] < t[child[0]]]
        avail_last = parent[t[parent] <= t[last]]
        best_first = avail_first[sim[child[0], avail_first].argmax()]
        best_last = avail_last[sim[last, avail_last].argmax()]
        cos_origin_first, cos_origin_last = sim[child[0], pos[origin_vid]], sim[last, pos[origin_vid]]
        rank = df["rank"].to_numpy()
        cos_best_last = sim[last, best_last]
        if len(child) == 1:
            pattern = "single version"
        elif rank[best_last] > rank[best_first] and cos_best_last >= ORIGIN_THRESHOLD and cos_best_last > cos_origin_last:
            pattern = "follows relative's updates"
        elif cos_origin_first - cos_best_last > DRIFT_DELTA:
            pattern = "drifts away"
        else:
            pattern = "stays close"
        rows.append(
            {
                "history_id": row.history_id,
                "repo": row.repo,
                "filename": row.filename,
                "scope": row.best_scope,
                "relative_repo": df.repo_full_name.iloc[pos[origin_vid]],
                "relative_filename": df.filename.iloc[pos[origin_vid]],
                "relative_history_id": origin_hist,
                "relative_versions": len(parent),
                "child_versions": len(child),
                "cos_first_to_origin": round(float(cos_origin_first), 4),
                "cos_last_to_origin": round(float(cos_origin_last), 4),
                "cos_last_to_relative_then": round(float(sim[last, parent_then]), 4),
                "cos_last_to_best_relative": round(float(cos_best_last), 4),
                "matched_relative_rank_first": int(rank[best_first]),
                "matched_relative_rank_last": int(rank[best_last]),
                "pattern": pattern,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = prepare()
    vec = fit_tfidf(df.body)
    x = vec.transform(df.body)
    sim = (x @ x.T).toarray().astype(np.float32)

    versions, histories = within_history(df, sim)
    prior = prior_matches(df, sim)
    origins = origin_tracking(df, sim, prior)

    versions.to_csv(OUT_DIR / "version_evolution.csv", index=False)
    histories.to_csv(OUT_DIR / "history_evolution.csv", index=False)
    prior.to_csv(OUT_DIR / "prior_matches.csv", index=False)
    origins.to_csv(OUT_DIR / "origin_tracking.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    multi = histories[histories.n_versions > 1]
    print(f"versions={len(df)} histories={len(histories)} (with >1 version: {len(multi)})\n")
    print("== 1-2. Within-history evolution (histories with >1 version) ==")
    print(multi[["n_versions", "span_days", "n_frontmatter_only", "n_body_changes", "body_cos_first_last", "min_body_cos_first"]].describe().round(3).to_string())
    print("\nbody cosine first->last:", pd.cut(multi.body_cos_first_last, [0, 0.5, 0.7, 0.9, 0.99, 1.01], right=False).value_counts(sort=False).to_dict())
    added = versions.fm_keys_added.str.split("; ").explode().loc[lambda s: s.ne("") & s.notna()].value_counts().head(8)
    removed = versions.fm_keys_removed.str.split("; ").explode().loc[lambda s: s.ne("") & s.notna()].value_counts().head(8)
    print("frontmatter keys most often added:", added.to_dict())
    print("frontmatter keys most often removed:", removed.to_dict())
    print("transitions changing declared source:", int(versions.source_changed.fillna(False).astype(bool).sum()))

    print("\n== 3. Prior matches: first version of each file vs earlier versions of other files ==")
    firsts = prior.groupby("history_id").head(1)
    bins = [-0.001, 0.3, 0.5, 0.9, 0.99, 1.01]
    table = pd.DataFrame({s: pd.cut(firsts[f"{s}_cos"], bins, right=False).value_counts(sort=False) for s in SCOPES})
    table.loc["no earlier version in scope"] = [firsts[f"{s}_cos"].isna().sum() for s in SCOPES]
    print(table.to_string())
    print("best scope for first versions with best_cos >= 0.9:", firsts[firsts.best_cos >= 0.9].best_scope.value_counts().to_dict())

    print(f"\n== 4. Origin tracking (first-version prior match >= {ORIGIN_THRESHOLD}) ==")
    print(f"tracked files: {len(origins)}")
    print(pd.crosstab(origins.scope, origins.pattern, margins=True).to_string())


if __name__ == "__main__":
    main()
