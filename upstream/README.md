# Upstream templates

These are the GH-AW templates and shared components that GHAW-H files declare (`source:`) or import (remote `imports:`), fetched from GitHub. They are **not part of GHAW-H**; this study adds them to compare GHAW-H files against the templates they come from. The fetched data (`data/`) is committed, so the analysis can be reproduced even if a source repository changes or disappears. The git clones (`repos/`) are not committed; `fetch_upstream.py` re-creates them.

## How to reproduce

From the repository root:

```bash
uv run python upstream/fetch_upstream.py
```

The script needs GHAW-H in `ghaw-relations/data/raw/` and network access to github.com (anonymous HTTPS, no token). It takes about 30 s and is idempotent: existing clones are fetched again, not re-cloned.

## What it does

1. **Clones.** It makes blobless bare clones (`--filter=blob:none`) into `repos/`, which total about 90 MB. Commits and trees are local, and file contents are downloaded only for the files that are needed.
2. **Template histories** (`data/template_versions.parquet`) contain every add, modify or delete of a `.md` file under:
   - `githubnext/agentics`: `workflows/` (all templates and `shared/`)
   - `pulumi-labs/gh-aw-internal`: `.github/workflows/` and `.github/snippets/`
   - `github/gh-aw`: `.github/workflows/shared/`, plus the specific workflows referenced from GHAW-H
   - every other path referenced by a declared source or remote import

   The data comes from `git log --all --raw --no-renames`, one row per (path, commit). Frontmatter and body are split with the same rule GHAW-H uses; the split was checked against all 2,820 GHAW-H rows and matches every one.
3. **References** (`data/references.parquet`) contain every distinct `source:` value and remote import (`owner/repo/path@ref`) in GHAW-H, resolved to the exact file:
   - **Commit-SHA refs** resolve to that commit.
   - **Branch or tag refs** (5 values) resolve to the last commit on the branch *before the first GHAW-H version that uses it*.
   - `githubnext/gh-aw` is treated as an alias of `github/gh-aw` (the repo was renamed).
4. **Manifest** (`data/fetch_manifest.csv`) records the HEAD commit of each repo and the fetch time.

## Fetch of 2026-10-04 (UTC)

| Repo | HEAD | Template files | Template versions |
| --- | --- | ---: | ---: |
| githubnext/agentics | `5d11aa2a` | 215 | 1,364 |
| github/gh-aw | `54a96bb5` | 189 | 1,163 |
| pulumi-labs/gh-aw-internal | `fccb0509` | 7 | 57 |
| Alfresco/alfresco-build-tools | `c176315c` | 1 | 12 |
| githubnext/autoloop | `438782ff` | 0 | 0 |
| SonarSource/awesome-ai | unavailable (private or deleted) | – | – |

There are 2,596 template versions in total, and none is missing content.

### Reference resolution (172 distinct values)

| Kind | Resolved | Not resolved |
| --- | ---: | --- |
| `source:` (160) | 155 | 3 paths that **never existed** in `githubnext/agentics` (`issue-comment-handler.md`, `stale-closer.md`, `weekly-team-status.md`); 1 value with no path (`githubnext/autoloop`, used by 8 versions) |
| remote `imports:` (13) | 12 | 1 in `SonarSource/awesome-ai` (repo unavailable) |

### Sanity check

This compares each history's first non-stub version that declares a resolvable source (118 histories) with the template body at the declared ref:
- 94 bodies are **identical**.
- 6 have cosine in [0.9, 0.99), 11 in [0.5, 0.9), and 7 below 0.5.
- No frontmatter is identical, which is expected because adding a workflow writes the `source:` line into the frontmatter.

## Analysis: parent attribution

`attribute_parents.py` attributes each GHAW-H file to its most similar upstream template revision at the time. It also checks declared sources, measures the change at adoption, and tracks whether files follow template updates. It writes `results/version_parents.csv`, `results/history_parents.csv` and `results/peer_pairs.csv`, and runs in about 15 s:

```bash
uv run python upstream/attribute_parents.py
```


## Analysis: hypothesized scenarios (AI-written files, collages)

`detect_scenarios.py` looks for two scenarios the team proposed. It runs after `attribute_parents.py` (it reads `results/history_parents.csv`) and takes about 70 s:

```
uv run python upstream/detect_scenarios.py
```

- **Collage:** paragraph-level reuse from several earlier sources (templates or other GHAW-H files), with boilerplate filtered. Writes `results/collage_versions.csv` and `results/collage_sources.csv`.
- **AI-written:** two signals. `results/convergent_pairs.csv` holds files similar in meaning (mean-centered static embeddings, `minishlab/potion-base-32M` at revision `1e5a03f8eeb2c98b928fbbd846f22f816360919f`) but not in wording. `results/skeleton_files.csv` marks files that follow the body skeleton of gh-aw's creation-agent prompt, which exists since gh-aw commit `be57766ae` (2025-12-19).

`model2vec` and `regex` are dependencies of the project (`pyproject.toml`).

## Analysis: emoji tally

`emoji_tally.py` counts emojis (Unicode Extended_Pictographic, minus ©, ®, ™ and the arrows ↔–↙) in every GHAW-H version and upstream template revision. It records where they appear (frontmatter, headings, prose, code blocks) and, for template copies, whether each emoji was inherited from the template. It is a descriptive feature, not an AI detector. It runs after `attribute_parents.py` in about 10 s:

```
uv run python upstream/emoji_tally.py
```

It writes `results/emoji_versions.csv`, `results/emoji_histories.csv`, `results/emoji_templates.csv` and `results/emoji_inheritance.csv`.

## Columns

**`template_versions.parquet`**

| Column | Description |
| --- | --- |
| `repo` | Upstream repo, `owner/name` |
| `path` | File path in the upstream repo |
| `commit_sha` | Commit that added, modified or deleted the file |
| `committed_at` | Committer date of that commit |
| `change` | `A` (added), `M` (modified) or `D` (deleted; content is null) |
| `blob_sha` | Git blob of the new content |
| `content`, `frontmatter`, `body` | File content, split like GHAW-H |

**`references.parquet`**

| Column | Description |
| --- | --- |
| `kind` | `source` or `import` |
| `value` | The raw value as written in GHAW-H |
| `first_used_at` | First GHAW-H commit that uses the value |
| `n_versions_using` | Number of GHAW-H versions that use the value |
| `repo`, `path`, `ref` | Parsed parts of the value |
| `ref_type` | `commit` or `branch/tag` |
| `resolved_commit` | The commit the ref resolved to |
| `blob_sha` | Git blob of the file at that commit |
| `status` | `ok`, `path not found at ref`, `no path in value`, `repo not available` or `repo not fetched` |
| `content`, `frontmatter`, `body` | File content at the ref, split like GHAW-H |

## Licensing

This folder mixes our own work with third-party content:

| Path | License |
| --- | --- |
| `*.py` (our scripts) | CC0 1.0, like the rest of the repository |
| `results/` | Our computed values are CC0 1.0. They are derived from GHAW-H (CC-BY-4.0, attribution required) and from the templates below. Some columns name templates, files or emojis from them. |
| `data/template_versions.parquet`, `data/references.parquet` | **Third-party content under its original licenses**, listed below |
| `data/fetch_manifest.csv` | CC0 1.0 (repository names, commits, fetch dates) |
| `repos/` | not committed; re-created by `fetch_upstream.py` |

Content in `data/` comes only from these repositories, and keeps their licenses. The license texts are in `licenses/`, taken from each repository's `LICENSE` at HEAD on 2026-10-04:

| Repository | License | Copyright notice | File |
| --- | --- | --- | --- |
| `githubnext/agentics` | MIT | Copyright (c) 2025 GitHub | `licenses/githubnext_agentics.LICENSE` |
| `github/gh-aw` | MIT | Copyright GitHub, Inc. | `licenses/github_gh-aw.LICENSE` |
| `pulumi-labs/gh-aw-internal` | MIT | Copyright (c) 2026 pulumi-labs | `licenses/pulumi-labs_gh-aw-internal.LICENSE` |
| `Alfresco/alfresco-build-tools` | Apache-2.0 | (no NOTICE file in the repository) | `licenses/Alfresco_alfresco-build-tools.LICENSE` |

`githubnext/autoloop` has no license, and `SonarSource/awesome-ai` is unavailable. **No content from either is included**: they appear only as names in `references.parquet`, with an empty `content` column.

**Statement of changes** (Apache-2.0, section 4(b); also applies to the MIT files):
- Each file's content is stored unchanged in the `content` column.
- `frontmatter` and `body` are added as a split of that content at the YAML frontmatter delimiters, the same way GHAW-H splits its files.
- No file is otherwise modified. Files and revisions are selected as described above.
