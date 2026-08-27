# syntax=docker/dockerfile:1
# strands-robots dashboard — fleet cockpit container
# Build from repo root:  docker build -f strands_robots/dashboard/Dockerfile -t strands-robots-dashboard .

# ── stage 1: frontend ──────────────────────────────────────────────
FROM node:20-slim AS frontend
WORKDIR /fe
COPY strands_robots/dashboard/frontend/package*.json ./
RUN npm ci
COPY strands_robots/dashboard/frontend/ ./
RUN npm run build

# ── stage 2: runtime ───────────────────────────────────────────────
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libegl1 libosmesa6 libglib2.0-0 ffmpeg udev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY strands_robots/ strands_robots/
RUN pip install --no-cache-dir -e ".[sim-mujoco,mesh]" fastapi "uvicorn[standard]" websockets pyserial opencv-python-headless

COPY --from=frontend /fe/dist strands_robots/dashboard/frontend/dist

ENV STRANDS_MESH=true \
    STRANDS_MESH_MULTICAST=true \
    STRANDS_MESH_CAMERA_HZ=5 \
    STRANDS_ROBOTS_NO_DYLD_SHIM=1 \
    MUJOCO_GL=egl

EXPOSE 8080
CMD ["python", "-m", "strands_robots", "dashboard", "--port", "8080"]
