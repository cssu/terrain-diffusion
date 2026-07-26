#!/usr/bin/env bash
#
# Replies to `/test help` with the list of commands.

set -euo pipefail

body="$(
  cat << 'EOF'
**Commands available on this pull request**

| Comment | Description |
| --- | --- |
| `/test` or `/test all` | Every group below except `gpu`, at the same time |
| `/test python` | The quick tests. |
| `/test slow` | The tests marked `slow` |
| `/test gpu` | The tests marked `gpu`, on the project GPU machine |
| `/test web` | The 3D visualizer |
| `/test help` | Shows this message |

Anything after the group name is passed to pytest, so `/test python -k sampler`
runs the quick tests whose names contain "sampler".

These commands are for tests that are too slow or need hardware to run every time.
EOF
)"

gh api --method POST \
  "/repos/$GITHUB_REPOSITORY/issues/$ISSUE_NUMBER/comments" \
  -f body="$body"
