#!/usr/bin/env bash
#
# Installs the JavaScript packages for the visualizer in web/
#
# TODO: update when a web project is setup

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f web/package.json ]]; then
  echo "No web/package.json yet, so there is nothing to install."
  exit 0
fi

echo "Installing JavaScript packages in web/..."

if [[ -f web/pnpm-lock.yaml ]]; then
  corepack enable
  (cd web && pnpm install --frozen-lockfile)
elif [[ -f web/yarn.lock ]]; then
  corepack enable
  (cd web && yarn install --immutable)
elif [[ -f web/package-lock.json ]]; then
  (cd web && npm ci)
else
  (cd web && npm install)
fi
