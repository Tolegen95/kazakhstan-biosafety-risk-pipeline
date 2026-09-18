#!/usr/bin/env bash
# Start the API and the web UI locally, no Docker required.
set -euo pipefail
cd "$(dirname "$0")"

trap 'kill 0' EXIT INT TERM

python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8000 &
python3 -m http.server 8080 --directory web &

echo "API:  http://127.0.0.1:8000/docs"
echo "Web:  http://127.0.0.1:8080"
echo "(Ctrl+C stops both)"
wait
