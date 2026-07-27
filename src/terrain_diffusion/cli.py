"""Output and CLI

Overview

The command line entry point and the code that turns a finished height grid into files.

Neighbours and communication

- Takes a user request.
- Asks Generation Orchestration for the region.
- Writes the result to disk.

Run this file's main() with `uv run terrain-diffusion`. The command is wired up in
pyproject.toml under [project.scripts].
"""

import sys


def main() -> int:
    """Placeholder, so the command runs before any of the project is built."""
    print("terrain-diffusion is not built yet. See README.md for what goes here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
