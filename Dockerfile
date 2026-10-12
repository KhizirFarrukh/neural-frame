# NeuralFrame with NVIDIA GPU support: PyTorch (CUDA 13), TensorRT 10, ONNX Runtime, NVENC/NVDEC FFmpeg.
#
#   docker build -t neuralframe .
#   docker compose up                       # job service + web UI on http://127.0.0.1:8765
#   docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
#       -v "$PWD/data:/data" -v neuralframe-cache:/cache neuralframe upscale /data/in.mp4 -o /data/out.mkv
#
# Needs the NVIDIA Container Toolkit on the host. The "video" driver capability exposes NVENC/NVDEC.

ARG UBUNTU=24.04

# ---- FFmpeg: static GPL build with NVENC/NVDEC, libvmaf, x264/x265/SVT-AV1 ------------------------------
FROM ubuntu:${UBUNTU} AS ffmpeg
# Dated BtbN autobuild of FFmpeg 9.0.2 (the version NeuralFrame is developed against), checksum-pinned.
# Upstream prunes old autobuilds eventually: pass a newer FFMPEG_URL (and its FFMPEG_SHA256) if it is gone.
ARG FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-10-11-13-04/ffmpeg-n9.0.2-25-g67b60c310b-linux64-gpl-9.0.tar.xz
ARG FFMPEG_SHA256=73533d1ca1bc3d43ddf5b115b1b5736a5ffa1c9f25ef65d2d894b2d315b86bd5
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL "$FFMPEG_URL" -o /tmp/ffmpeg.tar.xz \
    && echo "$FFMPEG_SHA256  /tmp/ffmpeg.tar.xz" | sha256sum -c - \
    && mkdir -p /opt/ffmpeg && tar -xJf /tmp/ffmpeg.tar.xz -C /opt/ffmpeg --strip-components=1 \
    && rm /tmp/ffmpeg.tar.xz && /opt/ffmpeg/bin/ffmpeg -hide_banner -version | head -1

# ---- Python environment + native module -------------------------------------------------------------------
FROM ubuntu:${UBUNTU} AS build
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
# CUDA 13 PyTorch first, so the package install below does not pull a CPU build.
RUN pip install "torch>=2.9" --index-url https://download.pytorch.org/whl/cu130
WORKDIR /src
COPY pyproject.toml CMakeLists.txt README.md ./
COPY native native
COPY neuralframe neuralframe
RUN pip install ".[tensorrt,onnx-gpu]" && cd / && python -c "import neuralframe._native as n; print(n.build_info())"

# ---- runtime ---------------------------------------------------------------------------------------------
FROM ubuntu:${UBUNTU}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ffmpeg /opt/ffmpeg/bin/ffmpeg /opt/ffmpeg/bin/ffprobe /usr/local/bin/
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    NEURALFRAME_PATHS__CACHE_DIR=/cache \
    NEURALFRAME_API__HOST=0.0.0.0 \
    NEURALFRAME_API__INPUT_ROOTS='["/data"]' \
    NEURALFRAME_API__OUTPUT_ROOTS='["/data"]'
# Run unprivileged as the image's existing "ubuntu" user (UID 1000, usually the host user's UID, so files
# written to a bind-mounted ./data belong to you).
RUN mkdir -p /cache /data && chown ubuntu:ubuntu /cache /data
USER ubuntu
VOLUME ["/cache", "/data"]
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/v1/models', timeout=4)" || exit 1
ENTRYPOINT ["neuralframe"]
CMD ["serve"]
