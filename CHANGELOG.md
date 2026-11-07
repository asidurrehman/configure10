# Changelog

All notable changes to this project are documented here.

## [Unreleased]

Repository restructuring. **The optimisation algorithm is unchanged**: with the same random seed, results are identical to version 1.0.

### Added
- `config.yaml` for the output folder and run settings (population size, number of generations, mutation probability, number of parallel processes), with a `--config` option to use another file.
- `configure10_functions.py`: the NSGA-II helper functions, previously copied in both scripts, now shared in one module.
- Docstrings for all functions.
- Automated tests (`tests/`) and a GitHub Actions workflow (Ubuntu and Windows, Python 3.10 and 3.12).
- `CITATION.cff`, `pyproject.toml`, `requirements.txt`, `environment.yaml`, `.gitignore` and this changelog.

### Changed
- The default output folder is now `configure_output` inside the repository instead of `C:\configure`, so the demonstration runs on any operating system. Set `run_path` in `config.yaml` to use another folder.
- README rewritten with installation, configuration, usage, model-coupling and output documentation.

### Fixed
- Compatibility with NumPy 2.4 and newer (the plot axis limit now takes the single element of a one-element array explicitly; the value is the same).

## [1.0] - 2023-10-18

Version published with the article *Multi-objective optimisation framework for Blue-Green Infrastructure placement using detailed flood model*, Journal of Hydrology 638, 131571 (2024), https://doi.org/10.1016/j.jhydrol.2024.131571.
