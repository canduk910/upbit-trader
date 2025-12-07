#!/bin/sh
set -euo pipefail

# install requirements only if not already installed
if [ ! -d "/usr/local/lib/python3.11/site-packages" ]; then
  pip install --no-cache-dir -r /app/requirements.txt
fi

# execute the command provided via docker compose
exec "$@"

