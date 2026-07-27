#!/usr/bin/env bash
#
# Replies with the result of the tests once they have finished

set -euo pipefail

if [[ "$RESULT" == "success" ]]; then
  heading="Tests passed"
else
  heading="Tests did not pass ($RESULT)"
fi

summary=""
if [[ -n "${RESULTS_DIR:-}" && -d "${RESULTS_DIR:-}" ]]; then
  summary="$(python3 .github/scripts/summarise-results.py "$RESULTS_DIR")"
fi

body="$(
  cat << EOF
**$heading**

- **Groups run:** \`$TEST_GROUPS\`
- **Commit tested:** \`$SHORT_SHA\`
- **Full log:** [Actions run]($RUN_URL)

$summary
EOF
)"

gh api --method POST \
  "/repos/$GITHUB_REPOSITORY/issues/$ISSUE_NUMBER/comments" \
  -f body="$body"
