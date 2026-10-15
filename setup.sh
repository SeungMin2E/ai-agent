#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Run inside your RunPod Linux terminal, not local PowerShell." >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for candidate in python3 python3.12 python3.11 python3.13 python3.10; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; raise SystemExit(not ((3,10) <= sys.version_info[:2] <= (3,13)))'; then
    if [[ "$#" -eq 0 ]]; then set -- setup; fi
    exec "$candidate" "$script_dir/runpod_setup.py" "$@"
  fi
done
echo "Python 3.10-3.13 required. Use a RunPod GPU image with Python 3.12." >&2
exit 2
