#!/usr/bin/env bash
#
# Runs every check, should run before pushing.
#   scripts/format-check.sh     formatting and code style
#   scripts/test-check.sh       quick tests
#   scripts/web-check.sh        the visualizer, once web/ exists
#
# Run any of those on their own if you only want that part. To correct
# formatting problems, run scripts/format-check.sh --fix.

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

scripts/format-check.sh

echo
scripts/test-check.sh

echo
scripts/web-check.sh

echo
echo "All checks passed."
