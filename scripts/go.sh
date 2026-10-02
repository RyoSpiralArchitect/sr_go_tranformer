#!/bin/sh
# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
# Prefer the system Go; the local verified toolchain is a development fallback.
set -eu
cd "$(dirname "$0")/.."
if command -v go >/dev/null 2>&1; then
  exec go "$@"
fi
if [ -x .tools/go/bin/go ]; then
  exec .tools/go/bin/go "$@"
fi
printf '%s\n' 'Go 1.22+ is required: https://go.dev/dl/' >&2
exit 1
