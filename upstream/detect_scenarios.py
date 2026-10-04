"""Look for two hypothesized relation scenarios.

Run from the repository root after attribute_parents.py:

    uv run python upstream/detect_scenarios.py

Scenario A, "AI-written": someone asks an AI to write the instruction file. Such files
would look alike without ever being copies, because generation is probabilistic.
Two signals, reported separately:

1. convergent_pairs.csv (every pair down to the loosest sensitivity band, with `band` and
   `at_placeholder` columns): pairs of first versions (different files) that are similar in
   meaning (embedding cosine) but not in wording (word TF-IDF cosine below the gap),
   share no paragraph and are not siblings of one template family.
2. skeleton_files.csv: first versions that follow the body skeleton that gh-aw's own
   workflow-creation agent prompt (`create-agentic-workflow`) tells the AI to use:
   `## Your Task`, `## Guidelines`, `## Safe Outputs`, opening "You are an AI agent that".
   The skeleton first appeared in gh-aw on SKELETON_SINCE, inside the GHAW-H period,
   so it gives a before/after comparison.

Scenario B, "collage": someone copies paragraphs from several sources instead of one
whole body. Paragraphs (relations.fragments) are matched exactly or nearly (character
n-gram cosine) to earlier paragraphs in upstream templates and in other GHAW-H files.
Boilerplate (paragraphs found in several unrelated template families) is ignored.
A greedy cover then attributes the matched paragraphs to as few sources as possible.

3. collage_versions.csv: one row per first and last non-stub version of each file.
4. collage_sources.csv: one row per (version, contributing source).

All thresholds are placeholders, like those of the other scripts.
"""

import re
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from huggingface_hub import snapshot_download
from model2vec import StaticModel
from sklearn.feature_extraction.text import TfidfVectorizer

from attribute_parents import OUT_DIR, RELATED, load_templates, load_versions, template_families
from ghaw_relations.relations import fit_tfidf, fragments

EMBED_MODEL = "minishlab/potion-base-32M"
EMBED_REVISION = "1e5a03f8eeb2c98b928fbbd846f22f816360919f"

PARA_NEAR = 0.8  # placeholder: char n-gram cosine for two paragraphs to count as the same (edited) paragraph
BOILERPLATE_FAMILIES = 3  # placeholder: a paragraph found in this many upstream families is boilerplate
MIN_SOURCE_PARAS = 2  # placeholder: a source counts in a collage if it gives at least this many paragraphs
MIN_SOURCE_SHARE = 0.1  # placeholder: ... and at least this share of the file's paragraph words
WHOLE_COPY_SHARE = 0.8  # placeholder: one source covering this share is a whole copy, not a partial one
PARTIAL_MIN_SHARE = 0.33  # one source must cover this share to call it a partial copy (validation round 1, 2026-10-04)
LEX_LOW = 0.3  # placeholder: word TF-IDF cosine below the bimodal gap (exploration notes, section 2)
NEAR = 0.9  # placeholder: near-identical word cosine (same as attribute_parents)
# "Similar in meaning" is anchored on known relations instead of a fixed number: the median
# embedding cosine of sibling pairs (same template family) whose wording was adapted
# (word cosine in [LEX_LOW, NEAR)). Computed in main(); ANCHOR_QUANTILE is the placeholder.
ANCHOR_QUANTILE = 0.5
CJK_SHARE = 0.1  # files with more CJK characters than this: word TF-IDF can't split them into words
SKELETON_SINCE = pd.Timestamp("2025-12-19", tz="UTC")  # gh-aw be57766ae: skeleton added to the creation agent
SKELETON_HEADINGS = ("your task", "guidelines", "safe outputs")
SKELETON_OPENER = "you are an ai agent that"


# --- paragraphs -----------------------------------------------------------------------------------

def paragraph_index(df: pd.DataFrame, tmpl: pd.DataFrame) -> tuple[pd.DataFrame, list[list[int]]]:
    """Unique paragraphs of every non-stub GHAW-H version and upstream body revision,
    plus, for each paragraph id, the ids of its near-identical paragraphs (itself included)."""
    rows = [("v", i, f) for i, b in df.body[~df.stub].items() for f in dict.fromkeys(fragments(b))]
    rows += [("t", i, f) for i, b in tmpl.body.items() for f in dict.fromkeys(fragments(b))]
    occ = pd.DataFrame(rows, columns=["side", "row", "text"])
    texts = occ.text.drop_duplicates().reset_index(drop=True)
    occ["pid"] = occ.text.map(pd.Series(texts.index, index=texts))

    x = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(texts)
    nbrs: list[list[int]] = []
    for start in range(0, x.shape[0], 1000):
        block = (x[start:start + 1000] @ x.T).toarray()
        nbrs += [list(np.where(r >= PARA_NEAR)[0]) for r in block]
    return occ, nbrs


def paragraph_sources(occ, nbrs, df, tmpl, family):
    """Per paragraph id (its near-identical cluster): earliest time in each upstream family
    and in each GHAW-H history, and whether it is boilerplate."""
    t_up = tmpl.committed_at.to_numpy()
    t_gh = df.committed_at.to_numpy()
    hist = df.source_markdown_file_history_id.to_numpy()
    own_up: dict[int, dict] = defaultdict(dict)
    own_gh: dict[int, dict] = defaultdict(dict)
    for side, row, pid in occ[["side", "row", "pid"]].itertuples(index=False):
        d, key, t = (own_up[pid], family[row], t_up[row]) if side == "t" else (own_gh[pid], hist[row], t_gh[row])
        if key not in d or t < d[key]:
            d[key] = t
    up, gh = [], []
    for ns in nbrs:
        u, g = {}, {}
        for q in ns:
            for d_own, d_out in ((own_up.get(q, {}), u), (own_gh.get(q, {}), g)):
                for k, t in d_own.items():
                    if k not in d_out or t < d_out[k]:
                        d_out[k] = t
        up.append(u)
        gh.append(g)
    boiler = np.array([len(u) >= BOILERPLATE_FAMILIES for u in up])
    return up, gh, boiler


# --- scenario B: collage ---------------------------------------------------------------------------

def collage(df, targets, occ, up, gh, boiler, texts_words):
    meta = df.set_index("source_markdown_file_history_id")[["repo_full_name", "owner"]]
    meta = meta[~meta.index.duplicated()]
    by_row = occ[occ.side == "v"].groupby("row").pid.apply(list)
    v_rows, s_rows = [], []
    for i in targets:
        d = df.loc[i]
        pids = by_row.get(i, [])
        words = {p: texts_words[p] for p in pids}
        total = sum(words.values())
        cover = defaultdict(set)  # source -> paragraph ids it explains
        first_seen = {}
        n_boiler_words = 0
        for p in pids:
            if boiler[p]:
                n_boiler_words += words[p]
                continue
            for fam, t in up[p].items():
                if t <= d.committed_at:
                    cover[("template", fam)].add(p)
                    first_seen[("template", fam)] = min(t, first_seen.get(("template", fam), t))
            for h, t in gh[p].items():
                if h != d.source_markdown_file_history_id and t < d.committed_at:
                    cover[("ghaw", h)].add(p)
                    first_seen[("ghaw", h)] = min(t, first_seen.get(("ghaw", h), t))
        matched = set().union(*cover.values()) if cover else set()
        left, picked = set(matched), []
        while left:  # greedy cover: most uncovered words, then templates first, then earliest
            src = max(cover, key=lambda s: (sum(words[p] for p in cover[s] & left), s[0] == "template",
                                           -first_seen[s].value))
            got = cover[src] & left
            if not got:
                break
            picked.append((src, len(got), sum(words[p] for p in got)))
            left -= got
        sig = [s for s in picked if s[1] >= MIN_SOURCE_PARAS and s[2] / total >= MIN_SOURCE_SHARE]
        top_share = picked[0][2] / total if picked else 0.0
        if len(sig) >= 2:
            cat = "collage (2+ sources)"
        elif sig:
            cat = ("whole copy (1 source)" if top_share >= WHOLE_COPY_SHARE
                   else "partial copy (1 source)" if top_share >= PARTIAL_MIN_SHARE else "minor reuse")
        elif matched:
            cat = "minor reuse"
        else:
            cat = "no shared paragraphs"
        v_rows.append({
            "version_id": d.source_markdown_file_version_id, "history_id": d.source_markdown_file_history_id,
            "repo": d.repo_full_name, "owner": d.owner, "filename": d.filename, "rank": d["rank"],
            "which": d.which, "committed_at": d.committed_at, "n_paragraphs": len(pids), "paragraph_words": total,
            "boilerplate_share": round(n_boiler_words / total, 3) if total else np.nan,
            "matched_share": round(sum(words[p] for p in matched) / total, 3) if total else np.nan,
            "n_sources": len(picked), "n_significant_sources": len(sig), "top_source_share": round(top_share, 3),
            "significant_kinds": "+".join(sorted({scope(s[0], d, meta) for s in sig})), "category": cat,
        })
        for rank, (src, n, w) in enumerate(picked, 1):
            s_rows.append({
                "version_id": d.source_markdown_file_version_id, "which": d.which, "rank": rank,
                "kind": src[0], "source": src[1] if src[0] == "template" else meta.repo_full_name.get(src[1]),
                "source_history_id": src[1] if src[0] == "ghaw" else None, "scope": scope(src, d, meta),
                "paragraphs": n, "words": w, "share": round(w / total, 3), "significant": (src, n, w) in sig,
            })
    return pd.DataFrame(v_rows), pd.DataFrame(s_rows)


def scope(src, d, meta) -> str:
    if src[0] == "template":
        return "upstream template"
    repo, owner = meta.loc[src[1]]
    return "same repo" if repo == d.repo_full_name else "same owner" if owner == d.owner else "other owner"


# --- scenario A: AI-written -------------------------------------------------------------------------

def headings(body: str) -> list[str]:
    body = re.sub(r"(?ms)^(```|~~~).*?^\1", "", body)  # skip headings inside fenced code blocks
    out = []
    for h in re.findall(r"(?m)^#{1,6}\s+(.+?)\s*#*\s*$", body):
        h = re.sub(r"[^\w\s]", " ", h.lower())  # drops emoji and punctuation
        out.append(" ".join(h.split()))
    return out


def skeleton(body: str) -> dict:
    hs = set(headings(body))
    found = [h for h in SKELETON_HEADINGS if h in hs]
    opener = SKELETON_OPENER in " ".join(body.lower().split()[:60])
    return {"skeleton_headings": len(found), "skeleton_opener": opener,
            "skeleton_full": len(found) == len(SKELETON_HEADINGS)}


def embed(bodies: pd.Series) -> np.ndarray:
    path = snapshot_download(EMBED_MODEL, revision=EMBED_REVISION)
    e = StaticModel.from_pretrained(path).encode(bodies.tolist(), max_length=None)
    e = e / np.linalg.norm(e, axis=1, keepdims=True)
    e = e - e.mean(axis=0)  # remove the shared GH-AW domain component; raw cosines have a median of ~0.7
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def convergent_pairs(origins, lex, emb, shared_para, fam, emb_high):
    n = len(origins)
    iu = np.triu_indices(n, k=1)
    repo = origins.repo_full_name.to_numpy()
    owner = origins.owner.to_numpy()
    e, l, s = emb[iu], lex[iu], shared_para[iu]
    fa, fb = fam[iu[0]], fam[iu[1]]
    siblings = pd.notna(fa) & (fa == fb)
    cjk = origins.cjk.to_numpy()
    cjk_pair = cjk[iu[0]] | cjk[iu[1]]
    keep = (e >= emb_high) & (l < LEX_LOW) & ~s & ~siblings & ~cjk_pair
    a, b = iu[0][keep], iu[1][keep]
    out = pd.DataFrame({
        "history_a": origins.source_markdown_file_history_id.to_numpy()[a],
        "history_b": origins.source_markdown_file_history_id.to_numpy()[b],
        "repo_a": repo[a], "repo_b": repo[b],
        "file_a": origins.filename.to_numpy()[a], "file_b": origins.filename.to_numpy()[b],
        "embedding_cos": e[keep].round(4), "word_cos": l[keep].round(4),
        "skeleton_a": origins.skeleton_full.to_numpy()[a], "skeleton_b": origins.skeleton_full.to_numpy()[b],
    })
    out["scope"] = np.where(repo[a] == repo[b], "same repo", np.where(owner[a] == owner[b], "same owner", "other owner"))
    # baseline: how the same embedding band looks when wording is shared
    base = {
        "pairs": len(e), "emb>=high": int((e >= emb_high).sum()),
        "emb>=high & word>=low": int(((e >= emb_high) & (l >= LEX_LOW)).sum()),
        "emb>=high & word<low": int(((e >= emb_high) & (l < LEX_LOW)).sum()),
        "... & no shared paragraph & not siblings": int(((e >= emb_high) & (l < LEX_LOW) & ~s & ~siblings).sum()),
        "... & neither file mostly CJK (word cosine unreliable)": int(keep.sum()),
        "different repos only": int((keep & (repo[iu[0]] != repo[iu[1]])).sum()),
    }
    return out.sort_values("embedding_cos", ascending=False), base, (e, l)


def main() -> None:
    df = load_versions().reset_index(drop=True)
    tmpl = load_templates()
    vec = fit_tfidf(pd.concat([df.body, tmpl.body]))
    xt = vec.transform(tmpl.body)
    fam_map = template_families(tmpl, (xt @ xt.T).toarray())
    tmpl["family"] = tmpl.template.map(fam_map)

    hp = pd.read_csv(OUT_DIR / "history_parents.csv")
    hp["related_family"] = np.where(hp.parent_cos >= RELATED, hp.parent_family, None)
    hp["related_family"] = hp.related_family.where(hp.related_family.notna(), hp.declared_family)

    ns = df[~df.stub].sort_values("rank")
    g = ns.groupby("source_markdown_file_history_id")
    first, last = g.head(1).index, g.tail(1).index
    df["which"] = None
    df.loc[last, "which"] = "last"
    df.loc[first, "which"] = "first"  # a single-version file is reported as "first"

    # paragraphs
    occ, nbrs = paragraph_index(df, tmpl)
    texts_words = occ.drop_duplicates("pid").set_index("pid").text.str.split().str.len().sort_index().to_numpy()
    up, gh, boiler = paragraph_sources(occ, nbrs, df, tmpl, tmpl.family.to_numpy())
    print(f"paragraphs: unique={len(nbrs)} | boilerplate (in >= {BOILERPLATE_FAMILIES} upstream families)="
          f"{int(boiler.sum())}")

    # scenario B
    targets = df.index[df.which.notna()]
    cv, cs = collage(df, targets, occ, up, gh, boiler, texts_words)
    cv = cv.merge(hp[["history_id", "category"]].rename(columns={"category": "attribution"}), on="history_id", how="left")
    cv.to_csv(OUT_DIR / "collage_versions.csv", index=False)
    cs.to_csv(OUT_DIR / "collage_sources.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n== Scenario B: paragraph reuse (first and last non-stub version of each file) ==")
    print(pd.crosstab(cv.category, cv.which, margins=True).to_string(), "\n")
    col = cv[cv.category == "collage (2+ sources)"]
    print("collages by kind of significant sources:")
    print(pd.crosstab(col.significant_kinds, col.which).to_string(), "\n")
    print("collages by attribution category (first versions):")
    print(col[col.which == "first"].attribution.value_counts().to_string(), "\n")
    print(f"collage: median top-source share {col.top_source_share.median():.2f} "
          f"(base + additions if >= 0.5: {(col.top_source_share >= 0.5).sum()} of {len(col)})")
    became = cv.pivot_table(index="history_id", columns="which", values="category", aggfunc="first").dropna()
    print("files whose category changed from first to last version:")
    print(pd.crosstab(became["first"], became["last"]).to_string(), "\n")

    # scenario A
    origins = df.loc[first].reset_index()
    origins = origins.join(pd.DataFrame([skeleton(b) for b in origins.body]))
    origins["cjk"] = origins.body.map(
        lambda b: len(re.findall(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]", b)) / max(len(b), 1) > CJK_SHARE)
    origins = origins.merge(hp[["history_id", "category", "related_family"]],
                            left_on="source_markdown_file_history_id", right_on="history_id", how="left")
    xo = vec.transform(origins.body)
    lex = (xo @ xo.T).toarray()
    emb_m = embed(origins.body)
    emb = emb_m @ emb_m.T

    # shared non-boilerplate paragraph between two origins (exact or near)
    pid_rows = occ[(occ.side == "v") & occ.row.isin(origins["index"])]
    pid_rows = pid_rows[~boiler[pid_rows.pid.to_numpy()]]
    pos = pd.Series(np.arange(len(origins)), index=origins["index"])
    a = sp.csr_matrix((np.ones(len(pid_rows)), (pos[pid_rows.row].to_numpy(), pid_rows.pid.to_numpy())),
                      shape=(len(origins), len(nbrs)))
    nb_r = np.repeat(np.arange(len(nbrs)), [len(x) for x in nbrs])
    nb_c = np.concatenate(nbrs)
    near = sp.csr_matrix((np.ones(len(nb_r)), (nb_r, nb_c)), shape=(len(nbrs), len(nbrs)))
    shared_para = ((a @ near @ a.T) > 0).toarray()

    fam = origins.related_family.to_numpy()
    iu = np.triu_indices(len(origins), k=1)
    sib = pd.notna(fam[iu[0]]) & (fam[iu[0]] == fam[iu[1]])
    adapted = sib & (lex[iu] >= LEX_LOW) & (lex[iu] < NEAR)
    emb_high = float(np.quantile(emb[iu][adapted], ANCHOR_QUANTILE))
    print(f"\nanchor: {adapted.sum()} adapted sibling pairs, embedding cosine q25/q50/q75 = "
          f"{np.quantile(emb[iu][adapted], [.25, .5, .75]).round(3)} -> threshold {emb_high:.3f}")
    cp, base, (e_all, l_all) = convergent_pairs(origins, lex, emb, shared_para, fam, emb_high)
    # the CSV keeps every pair down to the loosest sensitivity threshold, labelled by band, for manual review
    bands = {q: float(np.quantile(emb[iu][adapted], q)) for q in (0.1, 0.25, 0.5, 0.75)}
    wide = convergent_pairs(origins, lex, emb, shared_para, fam, bands[0.1])[0]
    wide["band"] = wide.embedding_cos.map(lambda v: max(f"q{int(q * 100):02d}" for q, t in bands.items() if v >= t))
    wide["at_placeholder"] = wide.embedding_cos >= emb_high
    wide.to_csv(OUT_DIR / "convergent_pairs.csv", index=False)
    sk = origins[["source_markdown_file_history_id", "repo_full_name", "owner", "filename", "committed_at",
                  "category", "skeleton_headings", "skeleton_opener", "skeleton_full"]].rename(
        columns={"source_markdown_file_history_id": "history_id", "repo_full_name": "repo", "category": "attribution"})
    sk["period"] = np.where(sk.committed_at >= SKELETON_SINCE, "after skeleton", "before skeleton")
    sk.to_csv(OUT_DIR / "skeleton_files.csv", index=False)

    print("\n== Scenario A1: similar meaning, different wording (first versions, all pairs) ==")
    print(f"embedding cosine quantiles over all pairs: "
          f"{dict(zip(['q50', 'q90', 'q99', 'q999'], np.quantile(e_all, [.5, .9, .99, .999]).round(3)))}")
    print("same-template siblings (word cos >= 0.9) embedding cosine: median "
          f"{np.median(e_all[l_all >= 0.9]):.3f}, q05 {np.quantile(e_all[l_all >= 0.9], .05):.3f}")
    for k, v in base.items():
        print(f"  {k}: {v}")
    if len(cp):
        print("convergent pairs by scope:", cp.scope.value_counts().to_dict())
        files = pd.concat([cp.history_a, cp.history_b]).nunique()
        print(f"files in at least one convergent pair: {files} of {len(origins)}")
        print("pairs where both follow the creator skeleton:", int((cp.skeleton_a & cp.skeleton_b).sum()))
        print(cp.head(15)[["repo_a", "file_a", "repo_b", "file_b", "embedding_cos", "word_cos", "scope"]].to_string(index=False))
    for q in (0.1, 0.25, 0.5, 0.75):
        t = float(np.quantile(emb[iu][adapted], q))
        c = convergent_pairs(origins, lex, emb, shared_para, fam, t)[0]
        print(f"  sensitivity anchor q{int(q * 100)} ({t:.3f}): convergent pairs={len(c)}, "
              f"files={pd.concat([c.history_a, c.history_b]).nunique()}, scopes={c.scope.value_counts().to_dict()}")

    print("\n== Scenario A2: gh-aw creator-agent skeleton (first versions) ==")
    print(pd.crosstab([sk.period], sk.skeleton_headings, margins=True).to_string(), "\n")
    print("full skeleton by attribution category and period:")
    print(pd.crosstab(sk.attribution, [sk.period, sk.skeleton_full]).to_string(), "\n")
    print("opener 'You are an AI agent that' by period:", sk.groupby("period").skeleton_opener.sum().to_dict())
    print("opener by attribution category:", sk[sk.skeleton_opener].attribution.value_counts().to_dict())
    print("heading counts among files with 1-2 skeleton headings:",
          pd.Series([h for b, k in zip(origins.body, sk.skeleton_headings) if 0 < k < 3
                     for h in SKELETON_HEADINGS if h in set(headings(b))]).value_counts().to_dict())
    up_sk = tmpl.assign(**pd.DataFrame([skeleton(b) for b in tmpl.body], index=tmpl.index))
    up_first = up_sk[up_sk.skeleton_full].groupby("template").committed_at.min().sort_values()
    print(f"upstream templates whose body ever has the full skeleton: {len(up_first)}; "
          f"earliest: {up_first.head(3).to_dict()}")


if __name__ == "__main__":
    main()
