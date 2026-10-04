# ghaw-relations

Replication package for a study of **relations between GitHub Agentic Workflows (GH-AW) instruction files**. The study asks how the natural-language `.md` workflow files (YAML frontmatter + Markdown body) relate to each other: across repositories, and across versions of the same file.

> Status: **Stage 2, partial replication package.** It implements the full preliminary RQ1 analysis on GHAW-H v0.1.2: template relations, cross-owner relations, paragraph reuse, evolution and a first manual validation. Thresholds are provisional (see Thresholds); a second coder and the remaining threshold checks are pending.

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
upstream/          upstream GH-AW templates (not in GHAW-H): fetch, attribution and scenario scripts, data, results
```

## Reproducing

Requirements: [uv](https://docs.astral.sh/uv/) and Python ≥ 3.12.

```bash
uv sync                                    # create the environment
uv run python scripts/download_data.py     # fetch GHAW-H tables into data/raw/ (checksum-verified)
uv run python scripts/language_filter.py   # report repo languages; non-English/Spanish repos are excluded -> results/language/
uv run python scripts/flag_duplicates.py   # flag whitespace-only versions, reverts, forks, possible renames -> results/quality/
uv run python scripts/explore_relations.py # exploratory relation analysis -> results/exploration/
uv run python scripts/trace_evolution.py   # version-level evolution analysis -> results/evolution/
uv run python upstream/attribute_parents.py # files vs. the upstream templates they relate to -> upstream/results/
uv run python upstream/detect_scenarios.py  # partial copies, collages, AI-prompt skeleton -> upstream/results/
uv run python upstream/emoji_tally.py       # emoji use by location, inherited or added -> upstream/results/
uv run python scripts/validation_agreement.py # manual validation: agreement and kappa vs. the machine -> results/validation/
uv run python scripts/rq1_numbers.py        # every reported RQ1 number + the data behind each table -> results/rq1/
```

| Step | Output | Description |
| --- | --- | --- |
| `language_filter.py` | `results/language/repo_languages.csv` | Language of each repository's instructions. `load_snapshots()` excludes repos that are mostly not English or Spanish (3 repos), so all later steps use 259 repos |
| `flag_duplicates.py` | `results/quality/*.csv` | Flags duplicates without removing rows: 4 whitespace-only versions, 65 reverts, 2 fork pairs, 1 possible rename |
| `explore_relations.py` | `results/exploration/*.csv` | Exact matches, body similarity, shared fragments, `source:` validation, version changes, shared imports |
| `upstream/fetch_upstream.py` | `upstream/data/*.parquet` | Fetches the templates GHAW-H files declare or import. The data is committed; rerun only to refresh (it downloads newer commits too) |
| `upstream/attribute_parents.py` | `upstream/results/` | Each file's closest earlier template, declared vs. undeclared copies, change at adoption, whether files follow template updates, siblings vs. direct copies |
| `upstream/detect_scenarios.py` | `upstream/results/` | Paragraph reuse (partial copy, collage), similar-meaning pairs, gh-aw creator-prompt skeleton |
| `upstream/emoji_tally.py` | `upstream/results/` | Emoji use by location, and whether copies inherited it from the template |
| `validation_agreement.py` | `results/validation/round1_*.csv` | Agreement between manual labels (`validation/round1/`) and the automatic classification: overall, Cohen's kappa, per stratum |
| `rq1_numbers.py` | `results/rq1/numbers.csv`, `results/rq1/table_*.csv` | Collects every number reported for RQ1, with denominator, description and source file, and the counts behind each table. Reads stored results only |
| `trace_evolution.py` | `results/evolution/*.csv` | Evolution within each file history, earlier relatives across files (same repo / same owner / other owner), whether copies follow or drift from their relative |

## Results for RQ1

RQ1 asks how GitHub Agentic Workflows instruction files are related to each other. Each result maps to its script and output; `results/rq1/numbers.csv` lists every reported number with its source file.

| Result | Produced by | Output |
| --- | --- | --- |
| Dataset, inclusion and exclusion (language filter, stubs, duplicate flags) | `language_filter.py`, `flag_duplicates.py` | `results/language/`, `results/quality/` |
| Where files come from: template relation, declared vs. undeclared copies | `upstream/attribute_parents.py` | `upstream/results/history_parents.csv` → `results/rq1/table_attribution.csv` |
| Change at adoption (identical, near-identical, adapted) and frontmatter keys changed | `upstream/attribute_parents.py` | `results/rq1/table_adoption_change.csv`, `table_frontmatter_keys_changed.csv` |
| Whether files follow template updates | `upstream/attribute_parents.py` | `results/rq1/table_template_updates.csv` |
| Cross-owner near-identical pairs: template siblings vs. direct copies | `upstream/attribute_parents.py`, `explore_relations.py` | `results/rq1/table_cross_owner_pairs.csv`, `table_nearest_neighbour.csv` |
| Paragraph reuse: partial copies, collages | `upstream/detect_scenarios.py` | `results/rq1/table_paragraph_reuse.csv` |
| Divergent relation: gh-aw creator-prompt skeleton; similar-meaning pairs | `upstream/detect_scenarios.py` | `results/rq1/table_skeleton.csv`, `table_convergent_bands.csv` |
| Evolution within a file and against earlier relatives | `explore_relations.py`, `trace_evolution.py` | `results/rq1/table_version_changes.csv`, `table_relative_tracking.csv` |
| Emoji use (descriptive) | `upstream/emoji_tally.py` | `results/rq1/table_emoji_by_attribution.csv` |
| Consistency check of the categories (manual validation) | `validation_agreement.py` | `results/validation/round1_agreement.csv`, `round1_by_stratum.csv` |

The tables and figures in the course document are made by the authors from these files.

## Implementation status

| Component | Status |
| --- | --- |
| Pinned, checksum-verified data download | done |
| Data loading (`ghaw_relations.data`) | done |
| Exploratory relation analysis (exact, near, fragment, declared source, imports) | done (exploratory) |
| Version-level evolution tracing | done (exploratory) |
| Language filter (English/Spanish only, for manual checking) | done |
| Upstream templates: fetch, attribution, scenarios, emoji tally | done (exploratory; relation cutoff checked by hand) |
| Calibrated relation classification | partial: relation cutoff checked by hand on a small sample (see Thresholds) |
| RQ1 numbers and table data (`results/rq1/`) | done (preliminary) |
| RQ1 tables and figures for the document | made by the authors from `results/rq1/` |

## Manual validation

`validation/round1/` holds the first round of manual validation (2026-10-04):
- **Sample:** `sample_cases.csv`, 87 detected relations drawn by stratum with seed 261004. Each row has the machine verdict.
- **Labels:** `labels_coder1.csv`, one coder's verdicts.

**Procedure:** the coder judged each case in a side-by-side comparison viewer, in blind mode, so the machine category was hidden. The coder's first verdict was recorded before the machine verdict was shown. The viewer itself is not part of this repository.

**Exclusion:** case C073 is excluded, because the viewer showed an emptied template revision for it (a display error).

**Result:** 78 of 85 compared cases agree with the machine (91.8%), Cohen's kappa 0.83.
- The machine verdicts in this sample follow the categories *before* the recalibration that this round motivated (see Thresholds).
- **Agreement between two human coders is still pending.**

## Thresholds

Similarity thresholds are provisional and **may still change**.

- **Relation cutoff (0.3):** the minimum body cosine for two files to count as related. It was lowered from 0.5 after a manual check of a small stratified sample (87 cases, one coder, 2026-10-04). In that check, adapted rewrites at cosine 0.36–0.40 were judged related and files at 0.21 or below were not. The new cutoff also sits in the gap of the bimodal nearest-neighbour distribution (0.2–0.3). It is `ORIGIN_THRESHOLD` in `scripts/trace_evolution.py` and `RELATED` in `upstream/attribute_parents.py`.
- **Partial copy (≥ 33% from one source)** and **identical = same normalized text** (`upstream/`) come from the same manual check.
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
| `upstream/data/*.parquet` | Third-party template content under its original licenses: MIT (agentics, gh-aw, Pulumi) and Apache-2.0 (Alfresco). License texts and the statement of changes are in [`upstream/licenses/`](upstream/licenses/) and [`upstream/README.md`](upstream/README.md#licensing). |
| `upstream/results/` | Like `results/`: CC0 for our contribution, credit GHAW-H and the template repositories. |

CC-BY-4.0 has no share-alike clause, so CC0 is allowed for our own work. Attribution to GHAW-H still applies to anything derived from it, whatever license we put on our part.
