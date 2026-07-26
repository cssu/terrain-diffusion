#!/usr/bin/env bash
#
# Puts a 👀 reaction on the comment that asked for the tests

set -euo pipefail

gh api --method POST \
  "/repos/$GITHUB_REPOSITORY/issues/comments/$COMMENT_ID/reactions" \
  -f content=eyes > /dev/null
