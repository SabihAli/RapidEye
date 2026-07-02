#!/usr/bin/env bash
# Optional helper: resolve YouTube URL to a direct stream URL via yt-dlp.
# Usage: ./scripts/relay_youtube.sh "https://www.youtube.com/watch?v=..."
set -euo pipefail

URL="${1:?YouTube URL required}"
yt-dlp -g -f "best[height<=1080]" "$URL"
