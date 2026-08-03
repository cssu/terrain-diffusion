"""Checks the package skeleton is installed and importable.

These tests do not check any terrain behaviour, there is none yet. They check
that `src/terrain_diffusion` is actually installed into the environment, so a
broken packaging setup is caught here rather than in someone's first import.
"""

import importlib

import pytest

COMPONENT_MODULES = [
    "terrain_diffusion.inference",
    "terrain_diffusion.encoding",
    "terrain_diffusion.pipeline",
    "terrain_diffusion.sampler",
    "terrain_diffusion.store",
    "terrain_diffusion.orchestration",
    "terrain_diffusion.service",
    "terrain_diffusion.cli",
    "terrain_diffusion.benchmark",
]


@pytest.mark.parametrize("name", COMPONENT_MODULES)
def test_component_module_imports(name: str) -> None:
    """Every component from the project plan has a module and it imports.

    If this fails, run `uv sync`. It installs the project into the environment.
    """
    assert importlib.import_module(name) is not None


def test_the_command_line_entry_point_runs(capsys) -> None:
    """`uv run terrain-diffusion` calls this, so it should return 0 and say something."""
    from terrain_diffusion.cli import main

    assert main() == 0
    assert capsys.readouterr().out.strip() != ""
