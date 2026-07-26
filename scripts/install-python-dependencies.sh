#!/usr/bin/env bash
#
# Installs the Python packages the project needs.

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v uv > /dev/null 2>&1; then
  echo "uv is not installed, and this project uses it to manage packages." >&2
  echo >&2
  echo "Install it with:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  echo >&2
  echo "Other options: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

# will fail when new packages are installed but not synced
echo "Installing Python packages with uv..."
uv sync --frozen
