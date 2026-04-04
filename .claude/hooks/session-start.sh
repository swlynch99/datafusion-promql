#!/bin/bash
set -euo pipefail

# Only run in Claude remote environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

echo '{"async": true, "asyncTimeout": 300000}'

rustup toolchain install nightly --component rustfmt --profile minimal
cargo build
