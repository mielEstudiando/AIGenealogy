"""Tally emoji use in GH-AW instruction files and upstream templates.

Run from the repository root after attribute_parents.py:

    uv run python upstream/emoji_tally.py

This is a descriptive feature, not an AI detector: humans seldom put emojis in instruction files,
but emojis are also inherited from templates (which may themselves be AI-written) and are often
instructions about the agent's output ("use emojis moderately"). So the tally records where each
emoji sits and, for template copies, whether it was inherited or added.

Writes to upstream/results/:
1. emoji_versions.csv: one row per GHAW-H version (stubs included): counts by location and the emojis found.
2. emoji_histories.csv: one row per file history: emoji use in the first and last version and in any version,
   plus its template attribution.
3. emoji_templates.csv: one row per upstream template body revision.
4. emoji_inheritance.csv: template-attributed first versions vs. the template revision they match:
   emojis inherited, added and dropped.
"""

import re

import pandas as pd
import regex

from attribute_parents import OUT_DIR, RELATED, load_templates
from ghaw_relations.data import load_snapshots

# Extended_Pictographic, minus symbols that people type as plain text: (c) (r) (tm) and the arrows U+2194-2199.
EMOJI = regex.compile(r"(?:(?![©®™↔-↙])\p{Extended_Pictographic})")
CODE = re.compile(r"(?ms)^(```|~~~).*?^\1")
HEADING = re.compile(r"(?m)^#{1,6}\s+.*$")


def tally(frontmatter: str | None, body: str | None) -> dict:
    fm, body = frontmatter or "", body or ""
    code = "".join(m.group(0) for m in CODE.finditer(body))
    prose = CODE.sub("", body)
    heads = "\n".join(HEADING.findall(prose))
    found = {
        "frontmatter": EMOJI.findall(fm),
        "heading": EMOJI.findall(heads),
        "prose": EMOJI.findall(HEADING.sub("", prose)),  # body text outside headings and code blocks
        "code": EMOJI.findall(code),
    }
    every = [e for v in found.values() for e in v]
    return {
        **{f"n_{k}": len(v) for k, v in found.items()},
        "n_total": len(every),
        "has_emoji": bool(every),
        "emojis": "".join(dict.fromkeys(every)),  # distinct, in order of appearance
        "fm_emoji_field": bool(re.search(r"(?m)^emoji:", fm)),
        "mentions_emoji": bool(re.search(r"emoji", prose, re.I)),  # e.g. "use emojis moderately"
        "body_words": len(body.split()),
    }


def main() -> None:
    df = load_snapshots()
    vp = pd.read_csv(OUT_DIR / "version_parents.csv")
    hp = pd.read_csv(OUT_DIR / "history_parents.csv")

    ev = pd.concat([df[["source_markdown_file_version_id", "source_markdown_file_history_id", "repo_full_name",
                        "path", "rank", "committed_at"]].reset_index(drop=True),
                    pd.DataFrame([tally(f, b) for f, b in zip(df.frontmatter, df.body)])], axis=1)
    ev = ev.rename(columns={"source_markdown_file_version_id": "version_id", "source_markdown_file_history_id": "history_id",
                            "repo_full_name": "repo"})
    ev = ev.merge(vp[["version_id", "stub"]], on="version_id", how="left")
    ev.to_csv(OUT_DIR / "emoji_versions.csv", index=False)

    g = ev.sort_values("rank").groupby("history_id")
    eh = pd.DataFrame({
        "repo": g.repo.first(), "path": g.path.first(), "n_versions": g.size(),
        "first_has_emoji": g.has_emoji.first(), "last_has_emoji": g.has_emoji.last(), "any_has_emoji": g.has_emoji.any(),
        "first_emojis": g.emojis.first(), "last_emojis": g.emojis.last(),
        "first_n_heading": g.n_heading.first(), "first_n_frontmatter": g.n_frontmatter.first(),
        "stub_only": g.stub.all(),
    }).reset_index()
    eh = eh.merge(hp[["history_id", "category"]], on="history_id", how="left")
    eh["change"] = (eh.first_has_emoji.map({True: "had", False: "none"}) + " -> "
                    + eh.last_has_emoji.map({True: "has", False: "none"}))
    eh.to_csv(OUT_DIR / "emoji_histories.csv", index=False)

    tmpl = load_templates()
    et = pd.concat([tmpl[["template", "repo", "path", "commit_sha", "committed_at", "revision", "component"]].reset_index(drop=True),
                    pd.DataFrame([tally(f, b) for f, b in zip(tmpl.frontmatter, tmpl.body)])], axis=1)
    et.to_csv(OUT_DIR / "emoji_templates.csv", index=False)

    # inheritance: first non-stub version vs. the template revision it matches
    first = vp[~vp.stub].sort_values("rank").groupby("history_id").head(1)
    first = first[first.parent_cos >= RELATED][["version_id", "history_id", "parent_template", "parent_commit"]]
    first = first.merge(ev[["version_id", "emojis"]], on="version_id")
    first = first.merge(et[["template", "commit_sha", "emojis"]].rename(columns={"template": "parent_template",
                                                                                 "commit_sha": "parent_commit",
                                                                                 "emojis": "template_emojis"}),
                        on=["parent_template", "parent_commit"], how="left")
    first["inherited"] = [("".join(e for e in a if e in b)) for a, b in zip(first.emojis, first.template_emojis.fillna(""))]
    first["added"] = [("".join(e for e in a if e not in b)) for a, b in zip(first.emojis, first.template_emojis.fillna(""))]
    first["dropped"] = [("".join(e for e in b if e not in a)) for a, b in zip(first.emojis, first.template_emojis.fillna(""))]
    first = first.merge(hp[["history_id", "category"]], on="history_id", how="left")
    first.to_csv(OUT_DIR / "emoji_inheritance.csv", index=False)

    # --- summary ---
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(f"versions: {len(ev)} | with any emoji: {ev.has_emoji.sum()} ({ev.has_emoji.mean():.0%})")
    print(f"files (histories): {len(eh)} | any version with emoji: {eh.any_has_emoji.sum()} ({eh.any_has_emoji.mean():.0%}) | "
          f"first version: {eh.first_has_emoji.sum()} | last version: {eh.last_has_emoji.sum()}")
    loc = ev[ev.has_emoji]
    print("versions with emoji, by location:", {k: int((loc[f'n_{k}'] > 0).sum()) for k in ("frontmatter", "heading", "prose", "code")})
    print("versions with an `emoji:` frontmatter field:", int(ev.fm_emoji_field.sum()))
    print("\nfiles by template attribution (any version has emoji):")
    print(eh.groupby(eh.category.fillna("?")).agg(files=("any_has_emoji", "size"), with_emoji=("any_has_emoji", "sum"),
                                                  share=("any_has_emoji", "mean")).round(2).to_string())
    print("\nemoji use from first to last version:", eh[eh.n_versions > 1].change.value_counts().to_dict())
    top = pd.Series(list("".join(ev.drop_duplicates("history_id", keep="last").emojis))).value_counts().head(15)
    print("most common emojis (latest version of each file):", top.to_dict())
    latest_t = et[~et.component].sort_values("revision").groupby("template").tail(1)
    print(f"\nupstream workflow templates (latest revision): {len(latest_t)} | with emoji: {latest_t.has_emoji.sum()} "
          f"({latest_t.has_emoji.mean():.0%}) | by repo: "
          f"{latest_t.groupby('repo').has_emoji.agg(['sum', 'size']).apply(lambda r: f'{r['sum']}/{r['size']}', axis=1).to_dict()}")
    print(f"\ntemplate-attributed first versions: {len(first)} | with emoji: {(first.emojis != '').sum()} | "
          f"all inherited: {((first.emojis != '') & (first.added == '')).sum()} | some added: {(first.added != '').sum()} | "
          f"template had emoji but copy dropped all: {((first.template_emojis.fillna('') != '') & (first.emojis == '')).sum()}")


if __name__ == "__main__":
    main()
