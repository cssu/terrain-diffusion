#!/usr/bin/env bash
#
# Runs the tests based on groups:
#   scripts/test-check.sh             the quick tests, same as `python` below
#   scripts/test-check.sh python      the quick tests
#   scripts/test-check.sh slow        the tests marked `slow`
#   scripts/test-check.sh gpu         the tests marked `gpu`
#   scripts/test-check.sh all         every test
#
# Anything after the group name is passed straight to pytest:
#   scripts/test-check.sh python -k sampler
#   scripts/test-check.sh all -x

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

group=python
case "${1:-}" in
  --all)
    group=all
    shift
    ;;
  python | slow | gpu | all)
    group="$1"
    shift
    ;;
  web)
    echo "The web tests are not pytest, so they are not run from here." >&2
    echo "Use: scripts/web-check.sh" >&2
    exit 2
    ;;
  "")
    # No arguments, so the default group stands. quality-check.sh calls it this way.
    ;;
  -* | */* | *.py)
    # pytest arguments with no group in front of them.
    echo "Name the test group before the pytest arguments:" >&2
    echo "  scripts/test-check.sh python $*" >&2
    exit 2
    ;;
  *)
    echo "Unknown test group: $1" >&2
    echo "Usage: scripts/test-check.sh [python|slow|gpu|all] [pytest arguments]" >&2
    exit 2
    ;;
esac

case "$group" in
  python) markers='not slow and not gpu' ;;
  slow) markers='slow' ;;
  gpu) markers='gpu' ;;
  all) markers='' ;;
esac

marker_args=()
if [[ -n "$markers" ]]; then
  marker_args=(-m "$markers")
fi

# report failing tests in formatted table
report_args=()
if [[ -n "${JUNIT_XML:-}" ]]; then
  mkdir -p "$(dirname "$JUNIT_XML")"
  report_args=(--junitxml="$JUNIT_XML" -o junit_family=xunit1)
fi

echo "==> Running the '$group' tests (pytest)"

status=0
uv run pytest "${marker_args[@]+"${marker_args[@]}"}" "${report_args[@]+"${report_args[@]}"}" "$@" || status=$?

if [[ $status -eq 5 ]]; then
  if [[ "$group" == "python" ]]; then
    # Fail if no tests ran
    echo >&2
    echo "No tests were found." >&2
    echo "CI Verification test should have run, check CI for errors." >&2
    exit 1
  fi

  echo
  echo "No tests in the '$group' group, so there was nothing to run."
  exit 0
fi

exit "$status"
