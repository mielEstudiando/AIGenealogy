# ghaw-relations

Replication package for a study of **relations between GitHub Agentic Workflows (GH-AW) instruction files**. The study asks how the natural-language `.md` workflow files (YAML frontmatter + Markdown body) relate to each other: across repositories, and across versions of the same file.

> Status: **Stage 2, work in progress.** This is a partial replication package. Only data acquisition and loading are implemented so far.

- GitHub: _TBD_
- Zenodo DOI (this version): _TBD_
- Repository version used for the submitted results: _TBD (tag or commit)_

## Research question

**RQ1.** How are GH-AW instruction files related to each other? Exact wording pending; see `docs/` once the methodology is defined.

## Data

The study uses **GHAW-H v0.1.2** (Valenzuela-Toledo, Kehrer & Panichella, 2026), CC-BY-4.0:

- Zenodo: https://doi.org/10.5281/zenodo.22084012
- Hugging Face (identical copy): https://huggingface.co/datasets/pavtch/GHAW-H

The dataset covers 262 repositories, 604 source-file histories and 2,820 source Markdown snapshots and versions. See [`docs/data_provenance.md`](docs/data_provenance.md) for how it is obtained and which tables are used.

## Layout

```
data/raw/          GHAW-H parquet tables (downloaded, not committed)
data/processed/    derived data produced by the analysis
docs/              data provenance and methodology notes
notebooks/         exploratory and analysis notebooks
results/           tables, metrics and figures data reported in the paper
scripts/           entry-point scripts (download, analysis steps)
src/ghaw_relations/  reusable code (data loading, similarity, ...)
```

## Reproducing

Requirements: [uv](https://docs.astral.sh/uv/) and Python ≥ 3.12.

```bash
uv sync                                    # create the environment
uv run python scripts/download_data.py     # fetch GHAW-H tables into data/raw/ (checksum-verified)
```

Analysis steps and the results they produce for RQ1 will be listed here as they are implemented.

## Implementation status

| Component | Status |
| --- | --- |
| Pinned, checksum-verified data download | done |
| Data loading (`ghaw_relations.data`) | done |
| Similarity computation (frontmatter / body) | pending |
| Relation classification | pending |
| RQ1 results (tables, figures) | pending |

## Citation of the dataset

> Valenzuela-Toledo, P., Kehrer, T., & Panichella, S. (2026). *GHAW-H: A Dataset of GitHub Agentic Workflow Histories* (Version v0.1.2) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.22084012
