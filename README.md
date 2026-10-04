# ghaw-relations

Replication package for a study of **relations between GitHub Agentic Workflows (GH-AW) instruction files**. The study asks how the natural-language `.md` workflow files (YAML frontmatter + Markdown body) relate to each other: across repositories, and across versions of the same file.

> Status: **Stage 2, work in progress.** This is a partial replication package. Only data acquisition and loading are implemented so far.

- GitHub: <https://github.com/mielEstudiando/AIGenealogy>
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
uv run python scripts/language_filter.py   # report repo languages; non-English/Spanish repos are excluded -> results/language/
uv run python scripts/explore_relations.py # exploratory relation analysis -> results/exploration/
uv run python scripts/trace_evolution.py   # version-level evolution analysis -> results/evolution/
```

| Step | Output | Description |
| --- | --- | --- |
| `language_filter.py` | `results/language/repo_languages.csv` | Language of each repository's instructions. `load_snapshots()` excludes repos that are mostly not English or Spanish (3 repos), so all later steps use 259 repos |
| `explore_relations.py` | `results/exploration/*.csv` | Exact matches, body similarity, shared fragments, `source:` validation, version changes, shared imports |
| `trace_evolution.py` | `results/evolution/*.csv` | Evolution within each file history, earlier relatives across files (same repo / same owner / other owner), whether copies follow or drift from their relative |

## Implementation status

| Component | Status |
| --- | --- |
| Pinned, checksum-verified data download | done |
| Data loading (`ghaw_relations.data`) | done |
| Exploratory relation analysis (exact, near, fragment, declared source, imports) | done (exploratory) |
| Version-level evolution tracing | done (exploratory) |
| Language filter (English/Spanish only, for manual checking) | done |
| Calibrated relation classification | partial: relation cutoff checked by hand on a small sample (see Thresholds) |
| RQ1 results (tables, figures) | pending |

## Thresholds

Similarity thresholds are provisional and **may still change**.

- **Relation cutoff (0.3):** the minimum body cosine for two files to count as related. It was lowered from 0.5 after a manual check of a small stratified sample (87 cases, one coder, 2026-10-04). In that check, adapted rewrites at cosine 0.36–0.40 were judged related and files at 0.21 or below were not. The new cutoff also sits in the gap of the bimodal nearest-neighbour distribution (0.2–0.3). In this repository it is `ORIGIN_THRESHOLD` in `scripts/trace_evolution.py`.
- **Other values (0.9 near-identical, 0.1 drift, 20-word stubs):** still unvalidated placeholders.

A second coder and a check against declared sources (`source:`) are planned before the thresholds are fixed.

## Citation of the dataset

> Valenzuela-Toledo, P., Kehrer, T., & Panichella, S. (2026). *GHAW-H: A Dataset of GitHub Agentic Workflow Histories* (Version v0.1.2) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.22084012

## License

Different parts of this repository are under different licenses:

| Path | License |
|------|---------|
| Code (`src/`, `scripts/`, `notebooks/`), docs and configuration | [CC0 1.0](LICENSE) |
| `results/`, `data/processed/` | Derived from GHAW-H, which is CC-BY-4.0. Our own contribution is CC0 1.0, but reuse must still credit GHAW-H. See [`results/LICENSE.md`](results/LICENSE.md). |
| `data/raw/` (not committed) | GHAW-H, CC-BY-4.0. Content from each source repository keeps its original license. |

CC-BY-4.0 has no share-alike clause, so CC0 is allowed for our own work. Attribution to GHAW-H still applies to anything derived from it, whatever license we put on our part.
