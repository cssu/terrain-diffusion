#!/usr/bin/env bash
#
# check if user has permission to run test command
#   needs to be, owner, member, or collaborator

set -euo pipefail

case "$ASSOCIATION" in
  OWNER | MEMBER | COLLABORATOR)
    echo "$COMMENTER is allowed to run commands ($ASSOCIATION)."
    ;;
  *)
    gh api --method POST \
      "/repos/$GITHUB_REPOSITORY/issues/$ISSUE_NUMBER/comments" \
      -f body="Sorry @$COMMENTER, only people with access to this repository can run \`/test\`."

    echo "$COMMENTER is not allowed to run commands ($ASSOCIATION)." >&2
    exit 1
    ;;
esac
