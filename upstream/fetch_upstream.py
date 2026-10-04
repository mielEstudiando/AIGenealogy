"""Fetch the upstream GH-AW templates that GHAW-H files declare or import, outside the dataset.

Run from the repository root:

    uv run python upstream/fetch_upstream.py

Steps:
1. Blobless bare clones of the upstream repos into upstream/repos/ (created or fetched).
2. Every version of the template Markdown files under the paths in TEMPLATE_PATHS,
   from `git log --raw` -> upstream/data/template_versions.parquet.
3. Every `source:` value and remote `imports:` entry in GHAW-H, resolved to the exact
   file at its ref -> upstream/data/references.parquet.
"""

import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ghaw_relations.data import load_snapshots
from ghaw_relations.relations import frontmatter_imports, frontmatter_source

ROOT = Path(__file__).resolve().parent
REPOS_DIR = ROOT / "repos"
DATA_DIR = ROOT / "data"

REPOS = [
    "githubnext/agentics",
    "github/gh-aw",
    "Alfresco/alfresco-build-tools",
    "pulumi-labs/gh-aw-internal",
    "githubnext/autoloop",
    "SonarSource/awesome-ai",
]
ALIASES = {"githubnext/gh-aw": "github/gh-aw"}  # renamed repo; GitHub redirects the old name

# Paths whose full Markdown history is extracted. gh-aw is large, so only the shared/
# components plus the specific workflows referenced from GHAW-H are taken from it.
TEMPLATE_PATHS = {
    "githubnext/agentics": ["workflows/"],
    "pulumi-labs/gh-aw-internal": [".github/workflows/", ".github/snippets/"],
    "github/gh-aw": [".github/workflows/shared/"],
    "Alfresco/alfresco-build-tools": [],
    "githubnext/autoloop": [],
    "SonarSource/awesome-ai": [],
}

REF_RE = re.compile(r"^(?P<owner>[^/@]+)/(?P<repo>[^/@]+)(?:/(?P<path>[^@]+))?(?:@(?P<ref>.+))?$")
FRONTMATTER_RE = re.compile(r"^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)(.*)$", re.S)


def repo_path(repo: str) -> Path:
    return REPOS_DIR / (repo.replace("/", "_") + ".git")


def git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo_path(repo)), *args], capture_output=True, text=True, check=True).stdout


def clone_or_fetch(repo: str) -> bool:
    """Returns False when the repo can't be cloned anonymously (private or deleted)."""
    path = repo_path(repo)
    env = os.environ | {"GIT_TERMINAL_PROMPT": "0"}
    if path.exists():
        cmd = ["git", "-C", str(path), "fetch", "-q", "--filter=blob:none", "origin", "+refs/heads/*:refs/heads/*", "+refs/tags/*:refs/tags/*"]
    else:
        cmd = ["git", "clone", "-q", "--bare", "--filter=blob:none", f"https://github.com/{repo}.git", str(path)]
    return subprocess.run(cmd, env=env, capture_output=True).returncode == 0


def split_frontmatter(content: str) -> tuple[str, str]:
    """Same split as GHAW-H's frontmatter/body columns (reproduces all 2,820 rows)."""
    m = FRONTMATTER_RE.match(content)
    return (m.group(1), m.group(2)) if m else ("", content)


def read_blobs(repo: str, oids: list[str]) -> dict[str, str]:
    """Read blobs, fetching missing ones from the promisor remote in one batch first."""
    oids = sorted(set(oids))
    if not oids:
        return {}
    # `git fetch <oid>...` lets a partial clone download the listed blobs in a single request
    for i in range(0, len(oids), 500):
        subprocess.run(
            ["git", "-C", str(repo_path(repo)), "-c", "fetch.negotiationAlgorithm=noop",
             "fetch", "-q", "--no-tags", "--no-write-fetch-head", "--filter=blob:none", "origin", *oids[i : i + 500]],
            capture_output=True, text=True,
        )
    out = subprocess.run(
        ["git", "-C", str(repo_path(repo)), "cat-file", "--batch"],
        input=("\n".join(oids) + "\n").encode(), capture_output=True, check=True,
    ).stdout
    blobs, pos = {}, 0
    for oid in oids:
        header_end = out.index(b"\n", pos)
        header = out[pos:header_end].decode().split()
        if header[1] == "missing":
            pos = header_end + 1
            continue
        size = int(header[2])
        blobs[oid] = out[header_end + 1 : header_end + 1 + size].decode("utf-8", errors="replace")
        pos = header_end + 1 + size + 1
    return blobs


def template_history(repo: str, pathspecs: list[str]) -> pd.DataFrame:
    """Every add/modify/delete of a .md file under the pathspecs, with its content."""
    if not pathspecs:
        return pd.DataFrame()
    log = git(repo, "log", "--all", "--raw", "--no-renames", "--no-abbrev", "--format=@@%H %cI", "--", *pathspecs)
    rows, commit, date = [], None, None
    for line in log.splitlines():
        if line.startswith("@@"):
            commit, date = line[2:].split(" ", 1)
        elif line.startswith(":"):
            meta, path = line.split("\t", 1)
            _, _, _, new_oid, status = meta.split()
            if path.endswith(".md"):
                rows.append({"repo": repo, "path": path, "commit_sha": commit, "committed_at": date, "change": status, "blob_sha": new_oid})
    df = pd.DataFrame(rows).drop_duplicates(["path", "commit_sha"])
    blobs = read_blobs(repo, df.loc[df.change != "D", "blob_sha"].tolist())
    df["content"] = df.blob_sha.map(blobs)
    df[["frontmatter", "body"]] = df.content.fillna("").map(split_frontmatter).tolist()
    return df


def references() -> pd.DataFrame:
    """Every declared source / remote import in GHAW-H, with when it was first used."""
    df = load_snapshots()
    df["committed_at"] = pd.to_datetime(df.committed_at, utc=True)
    rows = []
    for row in df.itertuples():
        src = frontmatter_source(row.frontmatter)
        if src:
            rows.append(("source", src, row.committed_at))
        for imp in frontmatter_imports(row.frontmatter):
            if "@" in imp and imp.count("/") >= 3:
                rows.append(("import", imp, row.committed_at))
    refs = pd.DataFrame(rows, columns=["kind", "value", "used_at"])
    refs = refs.groupby(["kind", "value"]).agg(first_used_at=("used_at", "min"), n_versions_using=("used_at", "size")).reset_index()
    parts = refs.value.str.extract(REF_RE)
    refs["repo"] = (parts.owner + "/" + parts.repo).replace(ALIASES)
    refs["path"], refs["ref"] = parts.path, parts.ref
    return refs


def resolve(refs: pd.DataFrame, available: list[str]) -> pd.DataFrame:
    out = []
    for repo, group in refs.groupby("repo"):
        if repo not in REPOS:
            out.append(group.assign(status="repo not fetched"))
            continue
        if repo not in available:
            out.append(group.assign(status="repo not available"))
            continue
        resolved = []
        for row in group.itertuples():
            rec = {"ref_type": None, "resolved_commit": None, "blob_sha": None, "status": None}
            if not isinstance(row.path, str):
                rec["status"] = "no path in value"
            else:
                ref = row.ref if isinstance(row.ref, str) else "HEAD"
                is_sha = bool(re.fullmatch(r"[0-9a-f]{40}", ref))
                rec["ref_type"] = "commit" if is_sha else "branch/tag"
                try:
                    if is_sha:
                        commit = git(repo, "rev-parse", "--verify", "-q", f"{ref}^{{commit}}").strip()
                    else:
                        # branch refs move: take the commit the branch pointed to when the file was first committed
                        before = row.first_used_at.isoformat()
                        commit = git(repo, "rev-list", "-1", f"--before={before}", ref).strip()
                except subprocess.CalledProcessError:
                    commit = ""
                if not commit:
                    rec["status"] = "ref not found"
                else:
                    rec["resolved_commit"] = commit
                    tree = git(repo, "ls-tree", commit, "--", row.path).split()
                    if len(tree) >= 3 and tree[1] == "blob":
                        rec["blob_sha"], rec["status"] = tree[2], "ok"
                    else:
                        rec["status"] = "path not found at ref"
            resolved.append(rec)
        group = pd.concat([group.reset_index(drop=True), pd.DataFrame(resolved)], axis=1)
        blobs = read_blobs(repo, group.blob_sha.dropna().tolist())
        group["content"] = group.blob_sha.map(blobs)
        out.append(group)
    refs = pd.concat(out, ignore_index=True)
    refs[["frontmatter", "body"]] = refs.content.fillna("").map(split_frontmatter).tolist()
    return refs


def main() -> None:
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    available = [repo for repo in REPOS if clone_or_fetch(repo)]
    print("unavailable repos:", sorted(set(REPOS) - set(available)) or "none")

    refs = references()
    # also extract the full history of every declared/imported path, so copies can be placed on the template's timeline
    extra = refs.dropna(subset=["path"]).groupby("repo").path.apply(lambda s: sorted(set(s))).to_dict()
    histories = []
    for repo in available:
        paths = sorted(set(TEMPLATE_PATHS[repo]) | set(extra.get(repo, [])))
        histories.append(template_history(repo, paths))
    templates = pd.concat(histories, ignore_index=True)
    refs = resolve(refs, available)

    templates.to_parquet(DATA_DIR / "template_versions.parquet", index=False)
    refs.to_parquet(DATA_DIR / "references.parquet", index=False)

    heads = {repo: git(repo, "rev-parse", "HEAD").strip() if repo in available else "unavailable" for repo in REPOS}
    pd.DataFrame({"repo": list(heads), "head_commit": list(heads.values()),
                  "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}).to_csv(DATA_DIR / "fetch_manifest.csv", index=False)

    print("template versions:", len(templates), "| missing content:", int(templates.content.isna().sum() - (templates.change == "D").sum()))
    print(templates.groupby("repo").agg(files=("path", "nunique"), versions=("commit_sha", "size")).to_string(), "\n")
    print("references:", len(refs))
    print(pd.crosstab([refs.kind, refs.repo], refs.status).to_string())


if __name__ == "__main__":
    main()
