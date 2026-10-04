"""Loading helpers for the GHAW-H tables stored in data/raw/."""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_table(name: str) -> pd.DataFrame:
    path = RAW_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `uv run python scripts/download_data.py` first")
    return pd.read_parquet(path)


def load_snapshots(readable_only: bool = True) -> pd.DataFrame:
    """One row per source Markdown file version, joined with its snapshot, history and repository.

    By default, repositories whose instructions are mostly not in English or Spanish are dropped
    (ghaw_relations.language), because the team can't read and check them."""
    snapshot = load_table("source_markdown_file_snapshot")
    version = load_table("source_markdown_file_version")
    repository = load_table("repository")[["repository_id", "repo_full_name"]]
    df = (
        version.merge(snapshot, on="source_markdown_file_snapshot_id", validate="one_to_one")
        .merge(repository, on="repository_id", how="left", validate="many_to_one")
        .sort_values(["source_markdown_file_history_id", "rank"])
        .reset_index(drop=True)
    )
    if readable_only:
        from ghaw_relations.language import repo_languages

        langs = repo_languages(df)
        df = df[~df.repo_full_name.isin(langs.repo_full_name[langs.excluded])].reset_index(drop=True)
    return df
