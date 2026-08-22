#!/usr/bin/env bash
#
# Checks that the Python code is formatted correctly and 
# has no style problems using ruff
#
#   scripts/format-check.sh          check, and fail if anything is wrong
#   scripts/format-check.sh --fix    correct what can be corrected

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "${1:-}" in
  --fix)
    echo "==> Fixing what can be fixed automatically (ruff check --fix)"
    uv run ruff check --fix .

    echo
    echo "==> Formatting (ruff format)"
    uv run ruff format .

    echo
    echo "Done. Anything left over has to be fixed by hand."
    exit 0
    ;;
  "") ;;
  *)
    echo "Unknown option: $1" >&2
    echo "Usage: scripts/format-check.sh [--fix]" >&2
    exit 2
    ;;
esac

check_format=()
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  check_format=(--output-format=github)
fi

echo "==> Checking formatting (ruff format --check)"
if ! uv run ruff format --check .; then
  echo
  echo "Some files are not formatted. Fix them with:" >&2
  echo "  scripts/format-check.sh --fix" >&2
  exit 1
fi

echo
echo "==> Checking code style (ruff check)"
if ! uv run ruff check ${check_format[@]+"${check_format[@]}"} .; then
  echo
  echo "Found style problems. Many can be fixed automatically with:" >&2
  echo "  scripts/format-check.sh --fix" >&2
  exit 1
fi

echo
echo "Formatting and style are fine."
