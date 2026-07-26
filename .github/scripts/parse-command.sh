#!/usr/bin/env bash
#
# Reads the comment and works out what the workflow should execute
#
#   /test                         run all tests
#   /test all                     run all tests
#   /test <python|slow|gpu|web>   run specific tests
#   /test help      the list of commands
#
# Extra args are passed to pytest
#
# It writes four values for the later jobs to read:
#
#   mode         none, help or tests
#   matrix       the groups to run, as the JSON a job matrix expects
#   groups       the same list in words, for the reply comment
#   pytest_args  extra arguments to hand to pytest

set -euo pipefail

UBUNTU='"ubuntu-latest"'
# TODO: set up real gpu machine
GPU_MACHINE='["self-hosted","gpu"]'

# only read the first line
first_line="$(printf '%s' "$BODY" | head -n 1 | tr -d '\r')"
command="$(printf '%s' "$first_line" | awk '{print $1}')"
rest="$(printf '%s' "$first_line" | sed -E 's|^[[:space:]]*/test[[:space:]]*||; s|[[:space:]]*$||')"

mode=none
groups=()
pytest_args=""

if [[ "$command" != "/test" ]]; then
  echo "Not a /test command, stopping here."
else
  first_word="$(printf '%s' "$rest" | awk '{print $1}')"
  case "$first_word" in
    help | all | python | slow | gpu | web)
      pytest_args="$(printf '%s' "$rest" | sed -E 's|^[^[:space:]]+[[:space:]]*||')"
      ;;
    *)
      first_word=all
      pytest_args="$rest"
      ;;
  esac

  case "$first_word" in
    help) mode=help ;;
    # TODO: add gpu back to this list once the machine from the TODO above is
    # registered, otherwise will hang for 24h
    all) mode=tests; groups=(python slow web) ;;
    *) mode=tests; groups=("$first_word") ;;
  esac

  # pass rest of line as pytest args
  if [[ -n "$pytest_args" ]] && ! printf '%s' "$pytest_args" | grep -Eq '^[A-Za-z0-9 _./=:@-]+$'; then
    echo "These arguments contain characters that are not allowed: $pytest_args" >&2
    exit 1
  fi
fi

# build a strategy.matrix for github actions to consume
entries=()
for group in ${groups[@]+"${groups[@]}"}; do
  case "$group" in
    gpu) runs_on="$GPU_MACHINE" ;;
    *) runs_on="$UBUNTU" ;;
  esac
  escaped="${runs_on//\"/\\\"}"
  entries+=("{\"group\":\"$group\",\"runs_on\":\"$escaped\"}")
done

matrix="{\"include\":[$(
  IFS=,
  echo "${entries[*]-}"
)]}"
group_list="$(
  IFS=,
  echo "${groups[*]-}"
)"
group_list="${group_list//,/, }"

echo "Mode '$mode', groups: ${group_list:-none}, pytest arguments: ${pytest_args:-none}"

{
  echo "mode=$mode"
  echo "matrix=$matrix"
  echo "groups=$group_list"
  echo "pytest_args=$pytest_args"
} >> "$GITHUB_OUTPUT"
