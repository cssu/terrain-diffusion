#!/usr/bin/env bash
#
# Installs everything the project needs, run once after cloning
#   scripts/install-python-dependencies.sh
#   scripts/install-web-dependencies.sh
#
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

scripts/install-python-dependencies.sh

echo
scripts/install-web-dependencies.sh

echo
echo "Dependencies installed."
