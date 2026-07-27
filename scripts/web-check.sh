#!/usr/bin/env bash
#
# runs lint + correctness tests under web/.
#
# TODO: verify this works when web folder is populated

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f web/package.json ]]; then
  echo "No web/package.json yet, so there is nothing to check."
  exit 0
fi

runner=(npm run)
if [[ -f web/pnpm-lock.yaml ]]; then
  runner=(pnpm run)
elif [[ -f web/yarn.lock ]]; then
  runner=(yarn)
fi

for script in lint test build; do
  if node -e "const p=require('./web/package.json'); process.exit(p.scripts && p.scripts['$script'] ? 0 : 1)" 2> /dev/null; then
    echo
    echo "==> Running '$script' in web/"
    (cd web && "${runner[@]}" "$script")
  else
    echo "web/package.json has no '$script' script, skipping."
  fi
done
