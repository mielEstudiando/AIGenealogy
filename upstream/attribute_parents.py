"""Attribute GHAW-H instruction files to upstream template parents.

Run from the repository root after fetch_upstream.py:

    uv run python upstream/attribute_parents.py

Writes CSVs to upstream/results/:

1. version_parents.csv: every GHAW-H version vs. the upstream template body revisions that
   existed when it was committed. Best match, how current that revision was, and the
   declared source when there is one.
2. history_parents.csv: one row per file history, based on its first non-stub version.
   Attribution category, change at adoption (body, and frontmatter ignoring `source:`),
   and how the file evolves relative to its parent template.
3. peer_pairs.csv: near-identical cross-owner pairs between GHAW-H files (first versions).
   Siblings of the same template family vs. direct copies with no upstream parent.

Template families group upstream templates whose body revisions are near-identical
(renamed templates such as agentics `repo-status.md` -> `daily-repo-status.md`).

Stub bodies (ghaw_relations.relations.is_stub) are excluded on both sides.
All thresholds are placeholders, consistent with the exploration scripts.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ghaw_relations.data import load_snapshots
from ghaw_relations.relations import fit_tfidf, frontmatter_source, is_stub, text_hash

FAMILY: dict[str, str] = {}  # template -> family, filled in main()

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results"

RELATED = 0.3  # minimum body cosine for a template relation; 0.5 until manual validation round 1 (2026-10-04)
NEAR = 0.9  # placeholder: near-identical
DRIFT_DELTA = 0.1  # placeholder: drop in cosine to the parent that counts as drifting away
COS_BANDS = [0, RELATED, NEAR, 1.0001]
COS_LABELS = ["rewritten (<0.3)", "adapted [0.3,0.9)", "near-identical (>=0.9)"]
IDENTICAL = "identical (same text)"  # same normalized body as the template revision (hash), not a cosine band


def strip_source(frontmatter: str) -> str:
    """Frontmatter without the `source:` line that `gh aw add` injects."""
    return re.sub(r"(?m)^source:.*\n?", "", frontmatter)


def fm_dict(frontmatter: str) -> dict | None:
    try:
        data = yaml.load(frontmatter, Loader=yaml.BaseLoader)
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    data.pop("source", None)
    return data


def load_templates() -> pd.DataFrame:
    """Upstream body revisions: per path, keep only versions whose body changed (stubs and deletions dropped)."""
    t = pd.read_parquet(DATA_DIR / "template_versions.parquet")
    t = t[(t.change != "D") & t.content.notna()].copy()
    t["committed_at"] = pd.to_datetime(t.committed_at, utc=True)
    t = t[~t.body.map(is_stub)].sort_values(["repo", "path", "committed_at"])
    t["body_hash"] = t.body.map(text_hash)
    t = t[t.body_hash.ne(t.groupby(["repo", "path"]).body_hash.shift())].copy()
    t["template"] = t.repo + "/" + t.path
    t["revision"] = t.groupby("template").cumcount() + 1
    t["n_revisions"] = t.groupby("template").revision.transform("max")
    t["component"] = t.path.str.contains(r"/shared/|/snippets/")
    return t.reset_index(drop=True)


def template_families(tmpl: pd.DataFrame, tsim: np.ndarray) -> dict[str, str]:
    """Union templates (not components) that have any pair of revisions with cosine >= NEAR.
    The family is named after its earliest template."""
    names = tmpl.template.to_numpy()
    parent = {t: t for t in set(names)}

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    workflow = ~tmpl.component.to_numpy()
    a, b = np.where(np.triu(tsim >= NEAR, k=1) & workflow[:, None] & workflow[None, :])
    for i, j in zip(a, b):
        if names[i] != names[j]:
            parent[find(names[i])] = find(names[j])
    first_seen = tmpl.groupby("template").committed_at.min()
    groups = pd.Series({t: find(t) for t in parent})
    label = groups.index.to_series().groupby(groups.values).apply(lambda ts: first_seen[ts].idxmin())
    return {t: label[root] for t, root in groups.items()}


def load_versions() -> pd.DataFrame:
    df = load_snapshots()
    df["owner"] = df.repo_full_name.str.split("/").str[0]
    df["filename"] = df.path.str.rsplit("/", n=1).str[-1]
    df["committed_at"] = pd.to_datetime(df.committed_at, utc=True)
    df["source"] = df.frontmatter.map(frontmatter_source)
    df["stub"] = df.body.map(is_stub)
    return df


def declared(df: pd.DataFrame) -> pd.DataFrame:
    """Resolved declared source per GHAW-H version (only `source:` references with status ok)."""
    refs = pd.read_parquet(DATA_DIR / "references.parquet")
    refs = refs[refs.kind == "source"][["value", "status", "repo", "path", "body", "frontmatter"]]
    refs = refs.rename(columns={"value": "source", "status": "declared_status", "body": "declared_body",
                                "frontmatter": "declared_frontmatter"})
    refs["declared_template"] = refs.repo + "/" + refs.path
    return df.merge(refs.drop(columns=["repo", "path"]), on="source", how="left")


def version_parents(df: pd.DataFrame, tmpl: pd.DataFrame, sim: np.ndarray, dsim: np.ndarray) -> pd.DataFrame:
    t_v = df.committed_at.dt.tz_convert(None).to_numpy()
    t_t = tmpl.committed_at.dt.tz_convert(None).to_numpy()
    available = t_t[None, :] <= t_v[:, None]
    s = np.where(available, sim, -1.0)
    j = s.argmax(axis=1)
    best = s[np.arange(len(df)), j]
    found = (best >= 0) & ~df.stub.to_numpy()

    # how current the matched revision was: latest revision of the same template available at that time
    latest_rev = np.zeros(len(df), dtype=int)
    tmpl_name = tmpl.template.to_numpy()
    for i in np.where(found)[0]:
        same = (tmpl_name == tmpl_name[j[i]]) & available[i]
        latest_rev[i] = tmpl.revision.to_numpy()[same].max()

    out = df[["source_markdown_file_version_id", "source_markdown_file_history_id", "repo_full_name", "owner",
              "filename", "rank", "committed_at", "stub", "source", "declared_status", "declared_template"]].copy()
    out.columns = ["version_id", "history_id", "repo", "owner", "filename", "rank", "committed_at", "stub", "source",
                   "declared_status", "declared_template"]
    pick = lambda col: np.where(found, tmpl[col].to_numpy()[j], None)  # noqa: E731
    out["parent_template"] = pick("template")
    out["parent_family"] = pick("family")
    out["parent_component"] = pick("component")
    out["parent_commit"] = pick("commit_sha")
    out["parent_committed_at"] = pick("committed_at")
    out["parent_revision"] = np.where(found, tmpl.revision.to_numpy()[j], np.nan)
    out["parent_latest_revision_then"] = np.where(found, latest_rev, np.nan)
    out["parent_cos"] = np.where(found, best.round(4), np.nan)
    out["declared_cos"] = np.where(~df.stub.to_numpy(), dsim.round(4), np.nan)
    return out


def classify_first(row) -> str:
    has_decl = isinstance(row.source, str)
    if row.stub_only:
        return "stub only (imports)"
    if has_decl and row.declared_status != "ok":
        return "declared, unresolvable"
    if has_decl:
        if row.declared_cos >= RELATED and row.parent_family == row.declared_family:
            return "declared, confirmed"
        if row.declared_cos >= RELATED:
            return "declared, confirmed (other template matches better)"
        return "declared, body diverged from declared template"
    if row.parent_cos >= RELATED:
        return "undeclared template copy" if not row.parent_component else "undeclared, matches a shared component"
    return "no upstream parent"


def history_parents(df: pd.DataFrame, vp: pd.DataFrame, tmpl: pd.DataFrame, sim: np.ndarray) -> pd.DataFrame:
    tpos = {t: np.where(tmpl.template.to_numpy() == t)[0] for t in tmpl.template.unique()}
    t_t = tmpl.committed_at.dt.tz_convert(None).to_numpy()
    rows = []
    for hid, g in vp.groupby("history_id", sort=False):
        g = g.sort_values("rank")
        ns = g[~g.stub]
        if ns.empty:
            first = g.iloc[0]
            rows.append({"history_id": hid, "repo": first.repo, "owner": first.owner, "filename": first.filename,
                         "n_versions": len(g), "stub_only": True, "source": first.source,
                         "declared_status": first.declared_status, "category": "stub only (imports)"})
            continue
        first, last = ns.iloc[0], ns.iloc[-1]
        i_first, i_last = first.name, last.name  # row positions in df / sim
        rec = {
            "history_id": hid, "repo": first.repo, "owner": first.owner, "filename": first.filename,
            "n_versions": len(g), "stub_only": False, "source": first.source, "declared_status": first.declared_status,
            "declared_template": first.declared_template, "declared_cos": first.declared_cos,
            "parent_template": first.parent_template, "parent_family": first.parent_family,
            "declared_family": FAMILY.get(first.declared_template, first.declared_template)
            if isinstance(first.declared_template, str) else None,
            "parent_component": first.parent_component,
            "parent_cos": first.parent_cos, "parent_revision": first.parent_revision,
            "identical_to_parent": isinstance(first.parent_template, str) and text_hash(df.loc[i_first].body) == tmpl.body_hash[
                (tmpl.template == first.parent_template) & (tmpl.revision == first.parent_revision)].iloc[0],
            "parent_latest_revision_then": first.parent_latest_revision_then,
            "adopted_latest_revision": first.parent_revision == first.parent_latest_revision_then,
            "days_after_parent_revision": round((first.committed_at - first.parent_committed_at).total_seconds() / 86400, 2)
            if first.parent_committed_at is not None else np.nan,
        }
        rec["category"] = classify_first(pd.Series(rec))

        # change at adoption: frontmatter vs the declared template's frontmatter, ignoring `source:`
        d = df.loc[i_first]
        if isinstance(d.declared_frontmatter, str) and d.declared_status == "ok":
            rec["fm_identical_ignoring_source"] = text_hash(strip_source(d.frontmatter)) == text_hash(strip_source(d.declared_frontmatter))
            a, b = fm_dict(d.frontmatter), fm_dict(d.declared_frontmatter)
            if a is not None and b is not None:
                keys = set(a) | set(b)
                rec["fm_keys_changed"] = "; ".join(sorted(k for k in keys if a.get(k) != b.get(k)))
                rec["fm_n_keys_changed"] = sum(a.get(k) != b.get(k) for k in keys)

        # evolution relative to the parent template (body revisions of that template only)
        if first.parent_cos >= RELATED and len(ns) > 1:
            p = tpos[first.parent_template]
            avail_last = p[t_t[p] <= last.committed_at.tz_convert(None).to_datetime64()]
            best_last = avail_last[sim[i_last, avail_last].argmax()]
            cos_orig_last = sim[i_last, tmpl.index[(tmpl.template == first.parent_template) & (tmpl.revision == first.parent_revision)][0]]
            cos_best_last = sim[i_last, best_last]
            latest_then = avail_last[-1]
            rec |= {
                "cos_last_to_adopted": round(float(cos_orig_last), 4),
                "cos_last_to_best_revision": round(float(cos_best_last), 4),
                "cos_last_to_latest_revision_then": round(float(sim[i_last, latest_then]), 4),
                "matched_revision_last": int(tmpl.revision[best_last]),
                "latest_revision_at_last": int(tmpl.revision[latest_then]),
                "template_updates_since_adoption": int(tmpl.revision[latest_then] - first.parent_revision),
            }
            if tmpl.revision[best_last] > first.parent_revision and cos_best_last >= RELATED and cos_best_last > cos_orig_last:
                rec["pattern"] = "follows template updates"
            elif first.parent_cos - cos_best_last > DRIFT_DELTA:
                rec["pattern"] = "drifts away"
            else:
                rec["pattern"] = "stays close"
        elif first.parent_cos >= RELATED:
            rec["pattern"] = "single non-stub version"
        rows.append(rec)
    hp = pd.DataFrame(rows)
    hp["adoption_change"] = pd.cut(hp.parent_cos, COS_BANDS, labels=COS_LABELS, right=False).astype(object)
    hp.loc[hp.identical_to_parent.fillna(False).astype(bool), "adoption_change"] = IDENTICAL
    return hp


def peer_pairs(df: pd.DataFrame, hp: pd.DataFrame, vsim: np.ndarray) -> pd.DataFrame:
    """Near-identical cross-owner first versions: shared template parent (siblings) or direct copy."""
    firsts = df[~df.stub].sort_values("rank").groupby("source_markdown_file_history_id").head(1)
    idx = firsts.index.to_numpy()
    s = vsim[np.ix_(idx, idx)]
    owner = firsts.owner.to_numpy()
    iu = np.triu_indices(len(idx), k=1)
    keep = (s[iu] >= NEAR) & (owner[iu[0]] != owner[iu[1]])
    a, b = iu[0][keep], iu[1][keep]
    par = hp.set_index("history_id")[["parent_family", "parent_cos"]]
    hid = firsts.source_markdown_file_history_id.to_numpy()
    out = pd.DataFrame({
        "history_a": hid[a], "history_b": hid[b],
        "repo_a": firsts.repo_full_name.to_numpy()[a], "repo_b": firsts.repo_full_name.to_numpy()[b],
        "file_a": firsts.filename.to_numpy()[a], "file_b": firsts.filename.to_numpy()[b],
        "cos": s[a, b].round(4),
    })
    for side in ("a", "b"):
        p = par.reindex(out[f"history_{side}"])
        out[f"parent_{side}"] = np.where(p.parent_cos.to_numpy() >= RELATED, p.parent_family.to_numpy(), None)
    # same repository name under different owners: almost always a fork
    out["likely_fork"] = out.repo_a.str.split("/").str[1].str.lower() == out.repo_b.str.split("/").str[1].str.lower()
    out["relation"] = np.select(
        [out.parent_a.notna() & (out.parent_a == out.parent_b), out.likely_fork,
         out.parent_a.isna() & out.parent_b.isna()],
        ["siblings (same template family)", "fork", "direct copy (no upstream parent)"],
        "mixed (different or one-sided parent)",
    )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = declared(load_versions())
    tmpl = load_templates()

    vec = fit_tfidf(pd.concat([df.body, tmpl.body]))
    xv, xt = vec.transform(df.body), vec.transform(tmpl.body)
    global FAMILY
    FAMILY = template_families(tmpl, (xt @ xt.T).toarray())
    tmpl["family"] = tmpl.template.map(FAMILY)
    sim = (xv @ xt.T).toarray()
    sim[df.stub.to_numpy()] = -1.0
    vsim = (xv @ xv.T).toarray()
    has_decl = df.declared_body.notna().to_numpy()
    dsim = np.full(len(df), np.nan)
    xd = vec.transform(df.declared_body[has_decl])
    dsim[has_decl] = np.asarray(xv[has_decl].multiply(xd).sum(axis=1)).ravel()

    vp = version_parents(df, tmpl, sim, dsim)
    hp = history_parents(df, vp, tmpl, sim)
    pp = peer_pairs(df, hp, vsim)

    vp.to_csv(OUT_DIR / "version_parents.csv", index=False)
    hp.to_csv(OUT_DIR / "history_parents.csv", index=False)
    pp.to_csv(OUT_DIR / "peer_pairs.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(f"GHAW-H versions={len(df)} histories={df.source_markdown_file_history_id.nunique()} | "
          f"upstream body revisions={len(tmpl)} templates={tmpl.template.nunique()} (components={tmpl[tmpl.component].template.nunique()})\n")
    print("== First-version attribution ==")
    print(hp.category.value_counts().to_string(), "\n")
    print(f"== Change at adoption (body cosine to parent, first non-stub version, parent cos >= {RELATED}) ==")
    print(pd.crosstab(hp.category, hp.adoption_change).loc[lambda x: x.sum(axis=1) > 0].to_string(), "\n")
    attributed = hp[hp.parent_cos >= RELATED]
    print(f"adopted the latest template revision available: {attributed.adopted_latest_revision.astype(bool).sum()} of {len(attributed)}; "
          f"median days after that revision: {attributed.days_after_parent_revision.median():.1f}")
    print("parent template families (top):", attributed.parent_family.value_counts().head(8).to_dict())
    multi = pd.Series(FAMILY).reset_index().groupby(0)["index"].apply(list).loc[lambda s: s.str.len() > 1]
    print(f"template families with >1 template (renames/copies upstream): {len(multi)}\n")
    if "fm_identical_ignoring_source" in hp:
        f = hp.dropna(subset=["fm_identical_ignoring_source"])
        print(f"frontmatter identical to declared template ignoring `source:`: {f.fm_identical_ignoring_source.sum()} of {len(f)}")
        keys = f.fm_keys_changed.dropna().str.split("; ").explode().loc[lambda s: s.ne("")].value_counts().head(10)
        print("keys most often changed at adoption:", keys.to_dict(), "\n")
    print("== Evolution relative to the parent template ==")
    print(hp.pattern.value_counts().to_string())
    ev = hp[hp.pattern.isin(["follows template updates", "drifts away", "stays close"])]
    print(f"template updates available by the file's last version: median {ev.template_updates_since_adoption.median()}, "
          f"files whose template changed at least once: {(ev.template_updates_since_adoption > 0).sum()} of {len(ev)}\n")
    print("== Near-identical cross-owner first versions ==")
    print(pp.relation.value_counts().to_string())


if __name__ == "__main__":
    main()
