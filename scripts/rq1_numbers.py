"""Collect every number reported for RQ1 into one file, plus the data behind each table.

    uv run python scripts/rq1_numbers.py

Run after the analysis scripts and scripts/validation_agreement.py (see README). It only reads stored results, plus GHAW-H for the
dataset counts, so the reported numbers can be traced and reproduced.

Writes to results/rq1/:
- numbers.csv: one row per reported number: section, key, value, denominator, share, description,
  source file.
- table_*.csv: the counts behind each table, in long or wide form. The tables and figures in the
  document are made by the authors from these files; this script does not format them.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ghaw_relations.data import load_snapshots
from ghaw_relations.relations import fit_tfidf, is_stub

ROOT = Path(__file__).resolve().parents[1]
RES, UP = ROOT / "results", ROOT / "upstream" / "results"
OUT = RES / "rq1"
H = "source_markdown_file_history_id"


class Numbers:
    def __init__(self):
        self.rows = []

    def add(self, section, key, value, description, source, denominator=None):
        value = value.item() if hasattr(value, "item") else value
        share = round(value / denominator, 4) if denominator else None
        self.rows.append({"section": section, "key": key, "value": value, "denominator": denominator,
                          "share": share, "description": description, "source": source})

    def frame(self):
        return pd.DataFrame(self.rows)


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def dataset(n: Numbers, df: pd.DataFrame, raw: pd.DataFrame) -> None:
    s = "dataset"
    langs = RES / "language" / "repo_languages.csv"
    lg = read(langs)
    n.add(s, "versions_raw", len(raw), "GHAW-H versions before the language filter", "GHAW-H v0.1.2")
    n.add(s, "repos_raw", raw.repo_full_name.nunique(), "repositories before the language filter", "GHAW-H v0.1.2")
    n.add(s, "repos_excluded_language", int(lg.excluded.sum()), "repositories excluded: mostly not English or Spanish", rel(langs))
    n.add(s, "versions", len(df), "versions analyzed", "GHAW-H v0.1.2")
    n.add(s, "files", df[H].nunique(), "files (histories) analyzed", "GHAW-H v0.1.2")
    n.add(s, "repos", df.repo_full_name.nunique(), "repositories analyzed", "GHAW-H v0.1.2")
    n.add(s, "owners", df.repo_full_name.str.split("/").str[0].nunique(), "owners (users or organizations)", "GHAW-H v0.1.2")
    vc = df.groupby(H).size()
    n.add(s, "files_multi_version", int((vc > 1).sum()), "files with more than one version", "GHAW-H v0.1.2", df[H].nunique())
    n.add(s, "first_commit", str(pd.to_datetime(df.committed_at, utc=True).min().date()), "earliest commit", "GHAW-H v0.1.2")
    n.add(s, "last_commit", str(pd.to_datetime(df.committed_at, utc=True).max().date()), "latest commit", "GHAW-H v0.1.2")
    stub = df.body.map(is_stub)
    n.add(s, "stub_versions", int(stub.sum()), "versions with a stub body (< 20 words)", "GHAW-H v0.1.2", len(df))
    n.add(s, "nonstub_files_latest", int((~df.sort_values("rank").groupby(H).body.last().map(is_stub)).sum()),
          "files whose latest version is not a stub", "GHAW-H v0.1.2")
    n.add(s, "declared_source_files", df[df.frontmatter.str.contains(r"(?m)^source:", regex=True)][H].nunique(),
          "files with a declared `source:` in some version", "GHAW-H v0.1.2", df[H].nunique())

    q = RES / "quality"
    vf, rf, rc = read(q / "version_flags.csv"), read(q / "repo_flags.csv"), read(q / "rename_candidates.csv")
    s = "data quality"
    n.add(s, "whitespace_only_versions", int(vf.whitespace_only.sum()), "versions identical to the previous one after normalization", rel(q / "version_flags.csv"), len(vf))
    n.add(s, "revert_versions", int(vf.revert_of_rank.notna().sum()), "versions returning to an earlier, non-consecutive version", rel(q / "version_flags.csv"), len(vf))
    n.add(s, "revert_files", vf[vf.revert_of_rank.notna()].history_id.nunique(), "files with at least one revert", rel(q / "version_flags.csv"))
    n.add(s, "fork_pairs", int(rf.fork.sum()) // 2, "repository pairs that are forks (same name, shared bodies)", rel(q / "repo_flags.csv"))
    n.add(s, "name_collision_pairs", int((~rf.fork).sum()) // 2, "same-name repository pairs with no shared content", rel(q / "repo_flags.csv"))
    n.add(s, "rename_candidates", len(rc), "possible renames within a repository", rel(q / "rename_candidates.csv"))


def attribution(n: Numbers) -> None:
    p = UP / "history_parents.csv"
    hp = read(p)
    s = "template relations"
    total = len(hp)
    t = hp.category.value_counts().rename_axis("category").reset_index(name="files")
    t["share"] = (t.files / total).round(4)
    t.to_csv(OUT / "table_attribution.csv", index=False)
    for cat, k in t.set_index("category").files.items():
        n.add(s, "first_version_" + cat.replace(" ", "_").replace(",", "").replace("(", "").replace(")", ""), k,
              f"first versions: {cat}", rel(p), total)
    related = hp[hp.parent_cos >= 0.3]
    n.add(s, "template_related_files", len(related), "files whose closest earlier template has body cosine >= 0.3 (confirmed and undeclared copies, plus declared sources that could not be resolved)", rel(p), total)

    ac = pd.crosstab(hp.category, hp.adoption_change)
    ac = ac.loc[[c for c in ("declared, confirmed", "undeclared template copy") if c in ac.index]]
    ac.to_csv(OUT / "table_adoption_change.csv")
    for cat, short in (("declared, confirmed", "declared"), ("undeclared template copy", "undeclared")):
        row = ac.loc[cat]
        for band, k in row.items():
            n.add(s, f"{short}_{band.split(' (')[0].split(' [')[0].replace('-', '_')}", k, f"{cat}: {band}", rel(p), int(row.sum()))

    fm = hp.dropna(subset=["fm_identical_ignoring_source"])
    n.add(s, "frontmatter_identical", int(fm.fm_identical_ignoring_source.astype(bool).sum()),
          "frontmatter identical to the declared template, ignoring `source:`", rel(p), len(fm))
    keys = fm.fm_keys_changed.dropna().str.split("; ").explode().loc[lambda x: x.ne("")].value_counts()
    keys.rename_axis("key").reset_index(name="files").to_csv(OUT / "table_frontmatter_keys_changed.csv", index=False)
    for k, v in keys.head(5).items():
        n.add(s, f"fm_key_changed_{k}", v, f"adoptions changing frontmatter key `{k}`", rel(p), len(fm))

    n.add(s, "adopted_latest_revision", int(related.adopted_latest_revision.astype(bool).sum()),
          "template-related files that adopted the latest revision available", rel(p), len(related))
    n.add(s, "median_days_after_revision", round(float(related.days_after_parent_revision.median()), 1),
          "median days between the template revision and its adoption", rel(p))

    ev = hp[hp.pattern.isin(["follows template updates", "stays close", "drifts away"])].copy()
    ev["template_changed"] = ev.template_updates_since_adoption > 0
    tu = pd.crosstab(ev.template_changed, ev.pattern)
    tu.to_csv(OUT / "table_template_updates.csv")
    ch = ev[ev.template_changed]
    for pat, k in ch.pattern.value_counts().items():
        n.add(s, "template_changed_" + pat.replace(" ", "_"), k, f"after the template changed: {pat}", rel(p), len(ch))
    fol = ch[ch.pattern == "follows template updates"]
    n.add(s, "follows_and_declared", int(fol.source.notna().sum()), "files following template updates that declare `source:`", rel(p), len(fol))


def cross_owner(n: Numbers) -> None:
    p = UP / "peer_pairs.csv"
    pp = read(p)
    s = "cross-owner relations"
    t = pp.relation.value_counts().rename_axis("relation").reset_index(name="pairs")
    t.to_csv(OUT / "table_cross_owner_pairs.csv", index=False)
    for rel_, k in t.set_index("relation").pairs.items():
        n.add(s, "pairs_" + rel_.split(" (")[0].replace(" ", "_"), k, f"near-identical cross-owner pairs: {rel_}", rel(p), len(pp))
    nn = RES / "exploration" / "nearest_neighbour_distribution.csv"
    read(nn).to_csv(OUT / "table_nearest_neighbour.csv", index=False)
    fn = read(RES / "exploration" / "file_nearest_neighbour.csv").dropna(subset=["max_cos_other_owner"])
    src = rel(RES / "exploration" / "file_nearest_neighbour.csv")
    n.add(s, "files_near_verbatim_other_owner", int((fn.max_cos_other_owner >= 0.99).sum()),
          "non-stub files with a near-verbatim (>= 0.99) file under another owner", src, len(fn))
    n.add(s, "files_related_other_owner", int((fn.max_cos_other_owner >= 0.3).sum()),
          "non-stub files with a related (>= 0.3) file under another owner", src, len(fn))
    n.add(s, "files_unrelated_other_owner", int((fn.max_cos_other_owner < 0.2).sum()),
          "non-stub files with nothing above 0.2 under another owner", src, len(fn))
    sv = read(RES / "exploration" / "source_validation.csv").set_index("pair_group")
    src = rel(RES / "exploration" / "source_validation.csv")
    n.add(s, "same_source_median_cos", sv.loc["same declared source", "q0.5"], "median body cosine, pairs with the same declared source", src)
    n.add(s, "diff_source_q95_cos", sv.loc["different declared source", "q0.95"], "95th percentile body cosine, pairs with different declared sources", src)


def evolution(n: Numbers) -> None:
    s = "evolution within a file"
    p = RES / "exploration" / "version_changes.csv"
    vc = read(p).dropna(subset=["change"])
    t = vc.groupby("change").agg(transitions=("change", "size"), median_body_cos=("body_cosine_to_previous", "median")).round(3)
    t.to_csv(OUT / "table_version_changes.csv")
    for c, k in t.transitions.items():
        n.add(s, "transitions_" + c.replace(" ", "_"), k, f"consecutive-version transitions: {c}", rel(p), int(t.transitions.sum()))
    p = RES / "evolution" / "history_evolution.csv"
    he = read(p)
    m = he[he.n_versions > 1]
    n.add(s, "files_never_change_body", int((m.n_body_changes == 0).sum()), "multi-version files whose body never changes", rel(p), len(m))
    n.add(s, "files_rewritten", int((m.body_cos_first_last < 0.3).sum()), "multi-version files rewritten (first-to-last cosine < 0.3)", rel(p), len(m))
    n.add(s, "median_versions", float(m.n_versions.median()), "median versions per multi-version file", rel(p))
    n.add(s, "median_span_days", float(m.span_days.median()), "median days between first and last version", rel(p))
    p = RES / "evolution" / "origin_tracking.csv"
    ot = read(p)
    t = pd.crosstab(ot.scope, ot.pattern)
    t.to_csv(OUT / "table_relative_tracking.csv")
    n.add(s, "tracked_files", len(ot), "files tracked against an earlier GHAW-H relative (cosine >= 0.3)", rel(p))
    for sc in ("same_owner", "other_owner"):
        d = ot[(ot.scope == sc) & (ot.pattern != "single version")]
        n.add(s, f"{sc}_follow", int((d.pattern == "follows relative's updates").sum()),
              f"{sc.replace('_', ' ')} copies following their relative's updates (multi-version)", rel(p), len(d))


def paragraphs(n: Numbers) -> None:
    s = "paragraph reuse"
    p = UP / "collage_versions.csv"
    cv = read(p)
    t = pd.crosstab(cv.category, cv.which)
    t.to_csv(OUT / "table_paragraph_reuse.csv")
    f = cv[cv.which == "first"]
    for c, k in f.category.value_counts().items():
        n.add(s, "first_" + c.split(" (")[0].replace(" ", "_"), k, f"non-stub first versions: {c}", rel(p), len(f))
    n.add(s, "collage_any", int((cv.category == "collage (2+ sources)").sum()), "first or last versions that are collages", rel(p), len(cv))


def generator(n: Numbers, df: pd.DataFrame) -> None:
    s = "divergent: creator-prompt skeleton"
    p = UP / "skeleton_files.csv"
    sk = read(p)
    t = pd.crosstab(sk.period, [sk.skeleton_opener, sk.skeleton_full])
    t.to_csv(OUT / "table_skeleton.csv")
    n.add(s, "first_versions", len(sk), "non-stub first versions checked", rel(p))
    n.add(s, "opener_files", int(sk.skeleton_opener.sum()), "files opening with 'You are an AI agent that'", rel(p), len(sk))
    n.add(s, "full_skeleton_files", int(sk.skeleton_full.sum()), "files with all three skeleton headings", rel(p), len(sk))
    n.add(s, "opener_before_skeleton", int(sk[sk.period == "before skeleton"].skeleton_opener.sum()),
          "opener files first committed before the skeleton existed (2025-12-19)", rel(p), int((sk.period == "before skeleton").sum()))
    n.add(s, "opener_no_template", int((sk.skeleton_opener & sk.attribution.eq("no upstream parent")).sum()),
          "opener files with no template relation", rel(p), int(sk.skeleton_opener.sum()))

    # similarity among files sharing the skeleton vs. independent files (cross-owner pairs of first versions)
    hp = read(UP / "history_parents.csv").set_index("history_id")
    ns = df[~df.body.map(is_stub)]
    first = ns.sort_values("rank").groupby(H).head(1).set_index(H)
    x = fit_tfidf(ns.body).transform(first.body)
    sim = (x @ x.T).toarray()
    owner = first.repo_full_name.str.split("/").str[0].to_numpy()
    opener = first.index.isin(sk[sk.skeleton_opener].history_id)
    indep = first.index.isin(hp[hp.category == "no upstream parent"].index) & ~opener

    def cross(mask):
        idx = np.where(mask)[0]
        a, b = np.triu_indices(len(idx), 1)
        a, b = idx[a], idx[b]
        keep = owner[a] != owner[b]
        return sim[a[keep], b[keep]]

    o, i = cross(opener), cross(indep)
    n.add(s, "opener_pair_median_cos", round(float(np.median(o)), 3), "median body cosine, cross-owner pairs of opener files", "computed from GHAW-H")
    n.add(s, "opener_pair_max_cos", round(float(o.max()), 3), "maximum body cosine, cross-owner pairs of opener files", "computed from GHAW-H")
    n.add(s, "independent_pair_median_cos", round(float(np.median(i)), 3), "median body cosine, cross-owner pairs of independent files", "computed from GHAW-H")

    p = UP / "convergent_pairs.csv"
    cp = read(p)
    cp.band.value_counts().rename_axis("band").reset_index(name="pairs").to_csv(OUT / "table_convergent_bands.csv", index=False)
    n.add(s, "convergent_pairs_placeholder", int(cp.at_placeholder.sum()), "similar-meaning, different-wording pairs at the placeholder cutoff", rel(p))


def emoji(n: Numbers) -> None:
    s = "emoji use"
    p = UP / "emoji_histories.csv"
    eh = read(p)
    n.add(s, "files_any_emoji", int(eh.any_has_emoji.sum()), "files with an emoji in some version", rel(p), len(eh))
    t = eh.groupby(eh.category.fillna("?")).agg(files=("any_has_emoji", "size"), with_emoji=("any_has_emoji", "sum"))
    t.to_csv(OUT / "table_emoji_by_attribution.csv")
    for c, r in t.iterrows():
        n.add(s, "emoji_" + c.replace(" ", "_").replace(",", "").replace("(", "").replace(")", ""), int(r.with_emoji),
              f"files with emoji: {c}", rel(p), int(r.files))
    p = UP / "emoji_versions.csv"
    ev = read(p)
    e = ev[ev.has_emoji]
    for loc in ("prose", "frontmatter", "code", "heading"):
        n.add(s, f"versions_emoji_in_{loc}", int((e[f"n_{loc}"] > 0).sum()), f"versions with an emoji in {loc}", rel(p), len(ev))
    p = UP / "emoji_inheritance.csv"
    ei = read(p).fillna({"emojis": "", "added": ""})
    w = ei[ei.emojis != ""]
    n.add(s, "copies_all_inherited", int((w.added == "").sum()), "template copies with emoji that inherited all of them", rel(p), len(w))


def validation(n: Numbers) -> None:
    s = "validation (round 1)"
    p = RES / "validation" / "round1_agreement.csv"
    if not p.exists():
        return
    a = read(p).iloc[0]
    n.add(s, "sample_cases", int(a.cases), "stratified sample of detected relations labeled by hand", rel(p))
    n.add(s, "coders", int(a.coders), "human coders (agreement is coder vs. machine)", rel(p))
    n.add(s, "compared", int(a.compared), "cases compared (excludes unsure and the display-error case)", rel(p))
    n.add(s, "agree", int(a.agree), "first verdicts agreeing with the machine", rel(p), int(a.compared))
    n.add(s, "cohen_kappa", float(a.cohen_kappa_vs_machine), "Cohen's kappa, coder vs. machine (binary related / not related)", rel(p))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_snapshots()
    n = Numbers()
    dataset(n, df, load_snapshots(readable_only=False))
    attribution(n)
    cross_owner(n)
    evolution(n)
    paragraphs(n)
    generator(n, df)
    emoji(n)
    validation(n)
    out = n.frame()
    out.to_csv(OUT / "numbers.csv", index=False)
    pd.set_option("display.width", 220, "display.max_colwidth", 70, "display.max_rows", 300)
    print(out[["section", "key", "value", "denominator", "share"]].to_string(index=False))
    print(f"\n{len(out)} numbers -> {rel(OUT / 'numbers.csv')}; tables: {sorted(x.name for x in OUT.glob('table_*.csv'))}")


if __name__ == "__main__":
    main()
