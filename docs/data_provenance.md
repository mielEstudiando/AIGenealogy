# Data provenance

## Source

- **Dataset:** GHAW-H: A Dataset of GitHub Agentic Workflow Histories, **v0.1.2**
- **Authors:** P. Valenzuela-Toledo, T. Kehrer, S. Panichella (University of Bern)
- **DOI:** [10.5281/zenodo.22084012](https://doi.org/10.5281/zenodo.22084012), published 2026-08-24
- **Mirror used for download:** Hugging Face `pavtch/GHAW-H`, pinned revision `9ccc840bd7e96fd90db3a16548d5482c2833c47d`. Zenodo lists it as `isIdenticalTo` the Zenodo record.
- **License:** CC-BY-4.0 for the dataset. Content from each source repository keeps its original license.
- **Companion docs:** https://github.com/pavt/GHAW-H, https://pavt.github.io/GHAW-H/

We do not re-run the extraction from GitHub. The analysis starts from the published GHAW-H tables.

## Acquisition

`scripts/download_data.py` downloads the tables from the pinned revision into `data/raw/` and checks each file's SHA-256. First download: 2026-10-03.

| Table | Rows | Used for |
| --- | ---: | --- |
| `repository` | 262 | Repository identity (`repo_full_name`) and metadata |
| `source_markdown_file_history` | 604 | Grouping versions into file histories |
| `source_markdown_file_snapshot` | 2,820 | File `content`, pre-split `frontmatter` and `body`, `path` |
| `source_markdown_file_version` | 2,820 | Commit SHA and timestamp, `rank` in history, predecessor and successor links |
| `lock_file_snapshot` (optional, `--with-lock`) | 2,820 | Compiled lock files. Not used for RQ1 at present |

The full column dictionary is in the [GHAW-H table dictionary](https://huggingface.co/datasets/pavtch/GHAW-H/blob/main/docs/table_dictionary.md).

## First observations (2026-10-03)

These come from `ghaw_relations.data.load_snapshots()`, which joins the version, snapshot and repository tables:

- 2,820 rows covering 262 repositories and 604 histories. Every snapshot has a frontmatter and a non-empty body.
- Commit timestamps range from 2025-10-03 to 2026-06-27.
- There are 1,287 distinct bodies (after trimming whitespace). 52 of them appear in more than one repository, and one body appears in 51 repositories.
- 302 rows have `content` identical to an earlier row.

## Transformations

None yet. Each transformation will be recorded here as it is added: normalization, deduplication, filtering, and so on.
