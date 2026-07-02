#!/usr/bin/env bash
# Vision Demo — start script (stub until Phase 1 is implemented)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "Missing .env — copy .env.example and set CAMERA_1_URL … CAMERA_4_URL"
  exit 1
fi

echo "Vision Demo is initialized but not yet implemented."
echo "See PLAN.md for implementation phases."
echo ""
echo "Planned usage:"
echo "  source .venv/bin/activate"
echo "  uvicorn server.main:app --host 0.0.0.0 --port 8000"
echo "  cd web && npm run dev"
exit 0
