"""Report the language of every repository and which ones the default load excludes.

    uv run python scripts/language_filter.py

Writes results/language/repo_languages.csv.
"""

from pathlib import Path

from ghaw_relations.data import load_snapshots
from ghaw_relations.language import MAX_OTHER_SHARE, READABLE, repo_languages

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "language"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_snapshots(readable_only=False)
    langs = repo_languages(df)
    langs.to_csv(OUT_DIR / "repo_languages.csv", index=False)
    ex = langs[langs.excluded]
    kept = df[~df.repo_full_name.isin(ex.repo_full_name)]
    print(f"readable languages: {READABLE}; exclude when other-language share > {MAX_OTHER_SHARE}")
    print(f"excluded repos: {len(ex)}")
    print(ex[["repo_full_name", "main_language", "other_share", "languages"]].to_string(index=False))
    print(f"kept: {kept.repo_full_name.nunique()} repos, {kept.source_markdown_file_history_id.nunique()} histories, "
          f"{len(kept)} versions (of {df.repo_full_name.nunique()}, {df.source_markdown_file_history_id.nunique()}, {len(df)})")
    mixed = langs[~langs.excluded & (langs.other_share > 0)]
    print("kept repos with some other-language text:")
    print(mixed[["repo_full_name", "other_share", "languages"]].to_string(index=False))


if __name__ == "__main__":
    main()
