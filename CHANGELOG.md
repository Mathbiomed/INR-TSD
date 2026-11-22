# Changes relative to the supplied new_INR_TSD.zip

## Documentation and release preparation — 2026-11-22

- Reorganized README around scope, installation, demonstration, configuration, custom input, reproducibility, and availability.
- Added Python 3.11 venv and conda alternatives and separate CPU, NVIDIA GPU and macOS dependency routes.
- Documented observation-dependent seasonal width and configurable trend initialization cycles, including their interpretation and limitations.
- Replaced outdated shared `N_HIDDEN_UNITS` documentation and unverified deterministic-execution claims.
- Added a validated custom-CSV example, focused TensorFlow tests, reproducibility notes, and an author release checklist.
- Corrected comments about the seasonal output mean and the main script command; added configuration comments. Numerical operations and configuration values in existing code are unchanged.
- Replaced the incorrectly named `gitignore` with `.gitignore`.
- Excluded caches, notebook checkpoints, .DS_Store, the transient weights and a misplaced historical output from this release copy. Original input CSVs, reference trend, figures and results are preserved in the uploaded source archive and retained here where appropriate.
- Retained the original MIT license pending author confirmation of the copyright-holder placeholder.

No full TensorFlow training, fresh dependency installation, GitHub publication or manuscript submission was performed during this update. Supplied figures and result CSVs are not new verification outputs.
