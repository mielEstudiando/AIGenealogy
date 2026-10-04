"""Exploratory analysis of relations between GH-AW instruction files (RQ1, Stage 2).

Writes CSVs to results/exploration/ and prints a summary. Five analyses:

1. Exact matches of content, frontmatter and body across all 2,820 versions.
2. Near matches: pairwise body similarity between files (latest version of each history).
3. Shared fragments: paragraphs reused across files, repositories and owners.
4. Validation against the `source:` frontmatter field (declared provenance).
5. Changes between consecutive versions of the same file.
6. Shared `imports:` targets (instructions composed from shared files).

Stub bodies (under STUB_MAX_WORDS words, typically a title whose instructions come from
`imports:`) are excluded from the body-similarity analyses (2 and 4) and reported separately.

Usage:
    uv run python scripts/explore_relations.py
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ghaw_relations.data import load_snapshots
from ghaw_relations.relations import (
    containment,
    fit_tfidf,
    fragments,
    frontmatter_keys,
    frontmatter_imports,
    frontmatter_source,
    is_stub,
    jaccard,
    text_hash,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "exploration"
PAIR_THRESHOLD = 0.3  # cross-repo file pairs with body cosine below this are not written out
COSINE_BINS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0001]


def prepare() -> pd.DataFrame:
    df = load_snapshots()
    df["owner"] = df.repo_full_name.str.split("/").str[0]
    df["filename"] = df.path.str.rsplit("/", n=1).str[-1]
    for col in ("content", "frontmatter", "body"):
        df[f"{col}_hash"] = df[col].map(text_hash)
    df["fm_keys"] = df.frontmatter.map(frontmatter_keys)
    df["source"] = df.frontmatter.map(frontmatter_source)
    # template = declared source without the @ref, e.g. githubnext/agentics/workflows/daily-repo-status.md
    df["source_template"] = df.source.str.replace(r"@.*$", "", regex=True)
    df["stub"] = df.body.map(is_stub)
    df["imports"] = df.frontmatter.map(frontmatter_imports)
    return df


def exact_matches(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Groups of versions with identical (whitespace-normalized) content, frontmatter or body."""
    groups, summary = [], []
    df = df.assign(body_nonstub_hash=df.body_hash.where(~df.stub))
    for level in ("content", "frontmatter", "body", "body_nonstub"):
        g = (
            df.dropna(subset=[f"{level}_hash"])
            .groupby(f"{level}_hash")
            .agg(
                n_versions=("source_markdown_file_version_id", "size"),
                n_histories=("source_markdown_file_history_id", "nunique"),
                n_repos=("repository_id", "nunique"),
                n_owners=("owner", "nunique"),
                filenames=("filename", lambda s: "; ".join(s.value_counts().index[:3])),
                owners=("owner", lambda s: "; ".join(s.value_counts().index[:3])),
            )
            .reset_index()
            .rename(columns={f"{level}_hash": "hash"})
        )
        g.insert(0, "level", level)
        groups.append(g[g.n_histories > 1])
        summary.append(
            {
                "level": level,
                "distinct": len(g),
                "groups_multi_history": int((g.n_histories > 1).sum()),
                "groups_multi_repo": int((g.n_repos > 1).sum()),
                "groups_multi_owner": int((g.n_owners > 1).sum()),
                "versions_in_multi_repo_groups": int(g.loc[g.n_repos > 1, "n_versions"].sum()),
                "histories_in_multi_repo_groups": int(
                    df[df[f"{level}_hash"].isin(g.loc[g.n_repos > 1, "hash"])].source_markdown_file_history_id.nunique()
                ),
                "largest_group_repos": int(g.n_repos.max()),
            }
        )
    groups = pd.concat(groups).sort_values(["level", "n_repos"], ascending=[True, False])
    return groups, pd.DataFrame(summary)


def file_pairs(files: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pairwise body similarity between files (one row per history: its latest version)."""
    tfidf = fit_tfidf(files.body).transform(files.body)
    sim = (tfidf @ tfidf.T).toarray()
    np.fill_diagonal(sim, np.nan)

    repo = files.repository_id.to_numpy()
    owner = files.owner.to_numpy()
    cross_repo = repo[:, None] != repo[None, :]
    cross_owner = owner[:, None] != owner[None, :]

    # nearest neighbour of each file in another repository / another owner
    nearest = files[["source_markdown_file_history_id", "repo_full_name", "filename"]].copy()
    nearest["max_cos_other_repo"] = np.nanmax(np.where(cross_repo, sim, -1), axis=1)
    nearest["max_cos_other_owner"] = np.nanmax(np.where(cross_owner, sim, -1), axis=1)
    nearest = nearest.replace(-1, np.nan)

    dist = (
        pd.DataFrame(
            {
                "files_by_max_cos_other_repo": pd.cut(nearest.max_cos_other_repo, COSINE_BINS, right=False).value_counts(sort=False),
                "files_by_max_cos_other_owner": pd.cut(nearest.max_cos_other_owner, COSINE_BINS, right=False).value_counts(sort=False),
            }
        )
        .rename_axis("cosine_bin")
        .reset_index()
    )
    dist["cosine_bin"] = dist.cosine_bin.astype(str)

    # cross-repo pairs above threshold, with fragment containment and frontmatter key overlap
    frags = [set(fragments(b)) for b in files.body]
    keys = files.fm_keys.tolist()
    tmpl = files.source_template.tolist()
    i_idx, j_idx = np.where(np.triu(cross_repo & (np.nan_to_num(sim) >= PAIR_THRESHOLD), k=1))
    pairs = pd.DataFrame(
        {
            "history_a": files.source_markdown_file_history_id.to_numpy()[i_idx],
            "history_b": files.source_markdown_file_history_id.to_numpy()[j_idx],
            "repo_a": files.repo_full_name.to_numpy()[i_idx],
            "repo_b": files.repo_full_name.to_numpy()[j_idx],
            "file_a": files.filename.to_numpy()[i_idx],
            "file_b": files.filename.to_numpy()[j_idx],
            "same_owner": ~cross_owner[i_idx, j_idx],
            "body_cosine": sim[i_idx, j_idx].round(4),
            "fragment_containment": [round(containment(frags[i], frags[j]), 4) for i, j in zip(i_idx, j_idx)],
            "fm_key_jaccard": [round(jaccard(keys[i], keys[j]), 4) for i, j in zip(i_idx, j_idx)],
            "same_declared_source": [
                isinstance(tmpl[i], str) and tmpl[i] == tmpl[j] for i, j in zip(i_idx, j_idx)
            ],
        }
    ).sort_values("body_cosine", ascending=False)
    return pairs, nearest, dist, sim


def shared_imports(files: pd.DataFrame) -> pd.DataFrame:
    """Import targets used by more than one file. Relative paths (e.g. shared/review.md) are resolved
    inside each repository, so the same name in two repos is not guaranteed to be the same content."""
    rows = files[["source_markdown_file_history_id", "repository_id", "owner", "imports"]].explode("imports").dropna()
    out = (
        rows.groupby("imports")
        .agg(n_files=("source_markdown_file_history_id", "nunique"), n_repos=("repository_id", "nunique"), n_owners=("owner", "nunique"))
        .reset_index()
        .rename(columns={"imports": "import_target"})
    )
    out["remote"] = out.import_target.str.count("/").ge(3) & out.import_target.str.contains("@")
    return out[out.n_files > 1].sort_values(["n_owners", "n_repos"], ascending=False)


def source_validation(files: pd.DataFrame, sim: np.ndarray) -> pd.DataFrame:
    """Body cosine of cross-repo pairs that declare the same source template vs. pairs that don't."""
    repo = files.repository_id.to_numpy()
    tmpl = files.source_template.to_numpy()
    has = pd.notna(tmpl)
    iu = np.triu_indices(len(files), k=1)
    cross = repo[iu[0]] != repo[iu[1]]
    both = has[iu[0]] & has[iu[1]]
    same = both & (tmpl[iu[0]] == tmpl[iu[1]])
    rows = []
    for label, mask in [
        ("same declared source", cross & same),
        ("different declared source", cross & both & ~same),
        ("no declared source (either side)", cross & ~both),
    ]:
        s = pd.Series(sim[iu][mask])
        rows.append({"pair_group": label, "n_pairs": len(s), **s.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(3).add_prefix("q").to_dict()})
    return pd.DataFrame(rows)


def shared_fragments(df: pd.DataFrame) -> pd.DataFrame:
    """Paragraph fragments that occur in more than one file history."""
    where = defaultdict(lambda: (set(), set(), set()))
    for row in df.itertuples():
        for frag in set(fragments(row.body)):
            h, r, o = where[frag]
            h.add(row.source_markdown_file_history_id)
            r.add(row.repository_id)
            o.add(row.owner)
    out = pd.DataFrame(
        [(f, len(h), len(r), len(o)) for f, (h, r, o) in where.items() if len(h) > 1],
        columns=["fragment", "n_histories", "n_repos", "n_owners"],
    )
    return out.sort_values(["n_owners", "n_repos"], ascending=False)


def version_changes(df: pd.DataFrame) -> pd.DataFrame:
    """What changes between consecutive versions of the same file."""
    tfidf = fit_tfidf(df.body).transform(df.body)
    pos = {vid: i for i, vid in enumerate(df.source_markdown_file_version_id)}
    rows = []
    for row in df.dropna(subset=["predecessor_source_markdown_file_version_id"]).itertuples():
        i, j = pos[int(row.predecessor_source_markdown_file_version_id)], pos[row.source_markdown_file_version_id]
        prev = df.iloc[i]
        fm_same, body_same = prev.frontmatter_hash == row.frontmatter_hash, prev.body_hash == row.body_hash
        kind = {
            (True, True): "no change",
            (False, True): "frontmatter only",
            (True, False): "body only",
            (False, False): "both",
        }[(fm_same, body_same)]
        rows.append(
            {
                "source_markdown_file_version_id": row.source_markdown_file_version_id,
                "history_id": row.source_markdown_file_history_id,
                "rank": row.rank,
                "change": kind,
                "body_cosine_to_previous": round(float((tfidf[i] @ tfidf[j].T).toarray()[0, 0]), 4),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = prepare()
    all_files = df.sort_values("rank").groupby("source_markdown_file_history_id").tail(1).reset_index(drop=True)
    files = all_files[~all_files.stub].reset_index(drop=True)

    groups, exact_summary = exact_matches(df)
    pairs, nearest, dist, sim = file_pairs(files)
    validation = source_validation(files, sim)
    frags = shared_fragments(df)
    changes = version_changes(df)
    imports = shared_imports(all_files)

    groups.to_csv(OUT_DIR / "exact_match_groups.csv", index=False)
    exact_summary.to_csv(OUT_DIR / "exact_match_summary.csv", index=False)
    pairs.to_csv(OUT_DIR / "file_pairs_cross_repo.csv", index=False)
    nearest.to_csv(OUT_DIR / "file_nearest_neighbour.csv", index=False)
    dist.to_csv(OUT_DIR / "nearest_neighbour_distribution.csv", index=False)
    validation.to_csv(OUT_DIR / "source_validation.csv", index=False)
    frags.to_csv(OUT_DIR / "shared_fragments.csv", index=False)
    changes.to_csv(OUT_DIR / "version_changes.csv", index=False)
    imports.to_csv(OUT_DIR / "shared_imports.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 80)
    print(f"versions={len(df)} files(histories)={len(all_files)} non-stub files={len(files)} repos={df.repository_id.nunique()} owners={df.owner.nunique()}")
    print(f"versions with declared source: {df.source.notna().sum()}  files: {all_files.source.notna().sum()}")
    print(f"stub bodies: {df.stub.sum()} versions, {all_files.stub.sum()} files (latest version); "
          f"{(df.stub & df.imports.str.len().gt(0)).sum()} stub versions have imports. \n")
    print("== 1. Exact matches (versions) ==\n", exact_summary.to_string(index=False), "\n")
    print("== 2. Nearest neighbour body cosine per file (latest version) ==\n", dist.to_string(index=False), "\n")
    print(f"cross-repo pairs with cosine >= {PAIR_THRESHOLD}: {len(pairs)} (same owner: {pairs.same_owner.sum()})\n")
    print("== 3. Shared fragments ==")
    print(f"fragments in >1 history: {len(frags)}; in >1 repo: {(frags.n_repos > 1).sum()}; in >1 owner: {(frags.n_owners > 1).sum()}")
    print(frags.head(5).assign(fragment=frags.fragment.str[:80]).to_string(index=False), "\n")
    print("== 4. Body cosine by declared source (cross-repo pairs) ==\n", validation.to_string(index=False), "\n")
    print("== 5. Consecutive version changes ==\n", changes.change.value_counts().to_string())
    print(changes.groupby("change").body_cosine_to_previous.describe().round(3).to_string(), "\n")
    print("== 6. Shared imports (latest version of each file) ==")
    print(f"files with imports: {all_files.imports.str.len().gt(0).sum()}; targets shared by >1 file: {len(imports)}, "
          f"by >1 owner: {(imports.n_owners > 1).sum()}, remote targets: {imports.remote.sum()}")
    print(imports.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
