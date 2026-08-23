#!/bin/bash

# Use GPU flags if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
  GPU_FLAGS="--gpus all"
else
  GPU_FLAGS=""
fi

xhost +

docker run $GPU_FLAGS -it --rm --privileged \
  -e DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e XAUTHORITY=/tmp/.docker.xauth \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /etc/localtime:/etc/localtime:ro \
  --network=host \
  iros2026/ai_module:latest
