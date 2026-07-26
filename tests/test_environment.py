"""Checks the development environment is set up correctly

Can maybe be deleted when real tests are added?
"""

import shutil
import sys

MINIMUM_PYTHON_VERSION = (3, 14)


def test_python_version_is_supported() -> None:
    """check project needs Python 3.14 or newer

    If this fails, run `uv sync`. uv reads the version from the
    .python-version file and will download the right one if it is missing.
    """
    assert sys.version_info[:2] >= MINIMUM_PYTHON_VERSION, (
        f"This project needs Python {MINIMUM_PYTHON_VERSION[0]}.{MINIMUM_PYTHON_VERSION[1]} "
        f"or newer, but the tests are running on {sys.version_info[0]}.{sys.version_info[1]}."
    )


def test_development_tools_are_installed() -> None:
    """ruff should be available as it's part of dev dependencies

    If this fails, run `uv sync` and `uv run pytest` so the project's environment is used
    """
    assert shutil.which("ruff") is not None, (
        "ruff was not found. Run `uv sync` to install the development "
        "dependencies, then run tests with `uv run pytest`."
    )


def test_slow_and_gpu_markers_are_registered(pytestconfig) -> None:
    """`slow` and `gpu` markers are registered in pyproject.toml.

    For now, `slow` and `gpu` are the only extra pytest markers that partition our tests
    """
    registered = {line.split(":", 1)[0] for line in pytestconfig.getini("markers")}

    for marker in ("slow", "gpu"):
        assert marker in registered, (
            f"The '{marker}' marker is not registered. Add it back to the "
            f"markers list in the [tool.pytest.ini_options] section of pyproject.toml."
        )
