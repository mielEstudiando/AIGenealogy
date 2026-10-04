# License of the results

The files in `results/` (and `data/processed/`) are derived from:

> Valenzuela-Toledo, P., Kehrer, T., & Panichella, S. (2026). *GHAW-H: A Dataset of GitHub Agentic Workflow Histories* (Version v0.1.2) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.22084012

GHAW-H is licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). We computed similarity scores, groupings and statistics from it, and we changed the data in this way. See `docs/data_provenance.md` for every transformation.

- **Our contribution** (the computed values and the table structure) is dedicated to the public domain under [CC0 1.0](../LICENSE).
- **Attribution:** if you reuse these files, cite GHAW-H as shown above. CC0 on our part does not remove that requirement.
- **Third-party content:** some columns contain text or identifiers from the original GitHub repositories, such as `shared_fragments.csv` (`fragment`), file and repository names, `declared_source` and `import_target`. That content keeps the license of its source repository.
