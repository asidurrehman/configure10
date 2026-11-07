"""Smoke tests for the configuration file and the two entry scripts."""
import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = ["configure10_standard.py", "configure10_parallelised.py"]


def test_default_config_has_expected_settings():
    with open(REPO / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config == {
        "run_path": "configure_output",
        "pop_size": 100,
        "generations": 3,
        "mutation_probability": 0.4,
        "processes_count": 4,
    }


@pytest.mark.parametrize("script", SCRIPTS + ["configure10_functions.py"])
def test_scripts_compile(script, tmp_path):
    py_compile.compile(str(REPO / script), cfile=str(tmp_path / "out.pyc"),
                       doraise=True)


@pytest.mark.parametrize("script", SCRIPTS)
def test_demo_runs_end_to_end(script, tmp_path):
    """Run a small demo optimisation and check the output files."""
    pop_size, generations = 20, 2
    config = {
        "run_path": "results",  # relative: resolved from the config's folder
        "pop_size": pop_size,
        "generations": generations,
        "mutation_probability": 0.4,
        "processes_count": 2,
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")

    env = dict(os.environ, MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, str(REPO / script), "--config", str(config_file)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "All done" in result.stdout

    out = tmp_path / "results"
    for name in ["generation_data.csv", "optimised_data.csv",
                 "BGI contribution to optimal solutions.csv",
                 "Gen_0.png", f"Gen_{generations}_optimal.png"]:
        assert (out / name).exists(), name

    data = pd.read_csv(out / "generation_data.csv")
    # one population per generation, including the initial one (generation 0)
    assert len(data) == pop_size * (generations + 1)
    assert sorted(data["Generation"].unique()) == list(range(generations + 1))
    assert {"Cost", "Risk", "BGI_0", "BGI_99"} <= set(data.columns)
