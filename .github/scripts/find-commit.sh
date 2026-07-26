#!/usr/bin/env bash
#
# find the latest commit of the branch a PR comment is made on

set -euo pipefail

sha="$(gh pr view "$ISSUE_NUMBER" \
  --repo "$GITHUB_REPOSITORY" \
  --json headRefOid \
  --jq .headRefOid)"

echo "Testing commit $sha"

{
  echo "sha=$sha"
  echo "short_sha=${sha:0:7}"
} >> "$GITHUB_OUTPUT"
