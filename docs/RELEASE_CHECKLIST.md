# Author checklist for a manuscript-associated release

## Required author information

- [ ] Replace `<Author Name>` in LICENSE with the verified copyright holder(s). Manuscript authorship alone does not establish ownership.
- [ ] Confirm the final manuscript title, authors and citation; add `CITATION.cff` using verified metadata.
- [ ] Add the actual GitHub repository URL to the README.
- [ ] Select a release tag, record its commit, and archive that release in a repository issuing a persistent identifier (for example, Zenodo). Add its DOI when issued; do not invent one.
- [ ] Record tested package versions, OS, hardware, driver information, and installation/demo timing from the actual manuscript environment.
- [ ] Add the source/provenance of `dataset/trend3.csv`.

## Scientific reproduction

- [ ] Install in a fresh Python 3.11 environment, run `pip check`, model tests, and the full demonstration.
- [ ] Verify CPU and GPU routes on the platforms claimed as tested; report only the platforms actually exercised.
- [ ] Confirm the scheduler behavior described in REPRODUCIBILITY.md matches the manuscript experiments.
- [ ] Confirm input-origin handling and the fixed-width sampling-degradation protocol for relevant experiments.
- [ ] Provide scripts and data/access instructions for biomedical, baseline, forecasting, and other manuscript experiments, or explicitly identify their separate repositories and restrictions.
- [ ] Map manuscript figures/tables to exact scripts, inputs, commands, configurations and expected metrics. The nine-series demo alone does not reproduce the whole paper.
- [ ] Archive reference output and environment metadata with the same source version used for submission.

## Draft code-availability statement

Complete this template only after the referenced resources exist. Extend its scope if additional analysis code is released:

> The INR-TSD core implementation and synthetic demonstration are available at [GITHUB_URL] under the MIT License. The version associated with this manuscript is [RELEASE_TAG / COMMIT], archived at [SOFTWARE_DOI]. Installation instructions, example inputs, model configuration, and verification procedures accompany the code. Scripts and access instructions for the remaining analyses are available at [ANALYSIS_CODE_AND_DATA_LOCATIONS], subject to [SPECIFY_ANY_ACCESS_RESTRICTIONS].

This template is not a completed availability statement. The placeholders identify author information absent from the supplied archive.

## Publishing the prepared files

Copy the contents of the prepared `INR-TSD/` directory into the intended repository, inspect the diff, run the verification commands, then commit and push through the normal GitHub workflow. Existing tracked caches or weights are not removed from Git history by adding `.gitignore`; review them explicitly in the destination repository.

The packaging update does not publish a GitHub repository, create a release, issue a DOI, or submit a manuscript. It preserves supplied input/output data while excluding disposable caches, notebook checkpoints, the ambiguously labelled weight pair, and a misplaced old decomposition output from the release copy. The original uploaded ZIP retains all excluded files.

## Guidance

- [Nature Communications submission policy](https://www.nature.com/ncomms/submit/policy)
- [Computational tools reproducibility guidelines](https://www.nature.com/documents/Computational_tools_reporting_guidelines.pdf)
- [TensorFlow installation](https://www.tensorflow.org/install/pip)
