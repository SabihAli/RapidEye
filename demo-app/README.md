# Vision Demo

Client-server security demo dashboard — **isolated from production RapidEye**.

- **Server:** camera ingest + GPU inference (FastAPI, Python)
- **Client:** local web UI for 4 live annotated camera tiles
- **Design spec:** [PLAN.md](PLAN.md)

## Separation from RapidEye

This folder is a self-contained subproject:

- Own Python virtualenv, Node dependencies, and runtime data under `data/`
- **No imports** from `vision-service/` or other RapidEye microservices
- Not deployed as part of the main platform

## Status

**Initialized only** — directory layout and plan are in place. Implementation follows [PLAN.md](PLAN.md) phases.

## Prerequisites (target dev machine)

| Component | Version |
|-----------|---------|
| OS | Ubuntu 22.04+ |
| GPU | NVIDIA RTX 3090 (CUDA 12.x) |
| Python | 3.10+ |
| Node.js | 20+ |
| System | `ffmpeg`, `yt-dlp` |

## Quick start (after implementation)

```bash
cd demo-app
cp .env.example .env
# Edit CAMERA_1_URL … CAMERA_4_URL (YouTube, RTSP, or HLS)

python -m venv .venv
source .venv/bin/activate
pip install -e .

cd web && npm install && cd ..
./scripts/start.sh
```

Open `http://localhost:5173` (dev) or `http://localhost:8000` (API + static build).

## Configuration

Copy `.env.example` to `.env`. Four independent stream URLs:

```
CAMERA_1_URL=...
CAMERA_2_URL=...
CAMERA_3_URL=...
CAMERA_4_URL=...
```

## Directory layout

```
demo-app/
  PLAN.md           # full design document
  server/           # FastAPI + inference pipeline (to be implemented)
  web/              # React dashboard (to be implemented)
  scripts/          # start helpers
  data/             # models, zones, recordings (gitignored)
```

## License

Internal demo — not for production use.
