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

## Transformations

Each transformation is listed here as it is added.

- **Joins:** `load_snapshots()` joins version → snapshot (1:1) → repository (n:1). `owner` is the part of `repo_full_name` before the `/`.
- **Exact matching:** whitespace normalization only. CRLF becomes LF, trailing spaces are stripped, runs of 3+ newlines become 2, and the text is trimmed. Then it's hashed with SHA-1 (`ghaw_relations.relations.normalize_text`).
- **Frontmatter parsing:** `yaml.BaseLoader`, so all values are strings and `on` stays a key rather than becoming `True`. One frontmatter fails to parse and gets null keys.
- **Fragments:** the body is split on blank lines, lowercased, and whitespace is collapsed. Fragments under 8 words are dropped.
- **Files:** each history is represented by its latest version (max `rank`) in `explore_relations.py`. `trace_evolution.py` uses every version.
- **Stub bodies:** bodies under 20 words (`ghaw_relations.relations.is_stub`) are flagged and excluded from body-similarity comparisons. Their instructions mostly come from `imports:` (176 of 193 stub versions).
- **Imports:** `imports:` entries are read as strings, or as the `path`/`uses`/first key of mapping entries.
- **TF-IDF:** word 1–2 grams with sublinear tf, fitted on the distinct bodies so that repeated versions don't skew the idf (`ghaw_relations.relations.fit_tfidf`).
- **Time:** `committed_at` is parsed as UTC. An "earlier" version must be strictly earlier, so same-timestamp ties are never treated as earlier.
- **Language filter (2026-10-04):** `load_snapshots()` drops, by default, repositories whose instructions the team can't read and check by hand (`ghaw_relations.language`). Each non-stub body is classified with `py3langid` after removing code blocks, inline code, `${{ ... }}` expressions, URLs and HTML tags or comments. A repository is excluded when more than 50% of its non-stub body words are in a language other than English or Spanish. Stubs have no prose, so they are ignored, and a repository with only stubs is kept. Excluded: `libxengine/XEngine_Authorize` (zh), `runhey/OnmyojiAutoScript` (zh) and `felipementel/GitHubCopilotDevDays-Curitiba-2026` (pt, 87%). That leaves **259 repos, 591 histories and 2,757 versions** of 262, 604 and 2,820. No version is in Spanish. The report is `scripts/language_filter.py` → `results/language/repo_languages.csv`. `load_snapshots(readable_only=False)` gives the unfiltered data.
- **Duplicates (team decision, 2026-10-04): no rows are removed; duplicates are flagged.** Repeated content across files is the phenomenon RQ1 studies, so it is never deduplicated.
  - **Against double counting:** file-level analyses use one version per file (usually the first non-stub version), and TF-IDF is fitted on distinct bodies. At record level, GHAW-H has no duplicated version, snapshot or (history, commit) rows.
  - **Flags,** from `scripts/flag_duplicates.py` → `results/quality/`:
    - **whitespace-only versions:** 4. Content is identical to the previous version after normalization. Already a separate "no change" transition.
    - **reverts:** 65 versions in 41 files return to an earlier, non-consecutive version; 36 are in `Azure/azure-sdk-for-js`. Kept as real history events.
    - **forks:** 2 pairs, `clash-verge-rev` and `oboapp`. Same repository name under two owners, with shared bodies. Reported separately in cross-owner results (`upstream/attribute_parents.py` labels them "fork"). 3 more same-name pairs share no content: name collisions, not forks.
    - **possible renames:** 1. `elastic/kibana` `reviewer-codex.md` ends 15.8 days before the near-identical `reviewer-scout.md` (cosine 0.91) starts. GHAW-H keeps one path per history and records no deletions, so a rename cannot be told apart from a new file derived from the old one.
- Apart from the language filter, no rows are removed.

## Upstream templates (not part of GHAW-H)

To relate GHAW-H files to the templates they come from, `upstream/fetch_upstream.py` fetches, from GitHub, the template repositories that GHAW-H files declare in `source:` or import remotely. These are `githubnext/agentics`, `github/gh-aw`, `pulumi-labs/gh-aw-internal` and `Alfresco/alfresco-build-tools`; `githubnext/autoloop` has no license and `SonarSource/awesome-ai` is unavailable.

- **Output:** `upstream/data/template_versions.parquet` (2,596 template versions) and `upstream/data/references.parquet` (resolved declared sources and remote imports). `upstream/data/fetch_manifest.csv` records each repository's head commit and fetch date.
- **Licensing:** the data is committed under its original licenses (see `upstream/README.md`, "Licensing").
- **Transformations:**
  - Each file's content is split into frontmatter and body exactly as GHAW-H does; the split reproduces GHAW-H's on all 2,820 rows.
  - The analyses use only *body revisions* (versions whose body changed), excluding deletions and stub bodies.
- **Details:** `upstream/README.md`.
