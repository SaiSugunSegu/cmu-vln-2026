FROM zhangjicmu/ubuntu24_ros:ai_module

# Install Foxglove bridge + system libs needed by OpenCV / captioner
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-foxglove-bridge \
      python3-pip \
      python3-yaml \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Captioner Python deps (torch CUDA + transformers / bitsandbytes). Layer is
# cached until requirements_captioner.txt changes.
COPY docker/requirements_captioner.txt /tmp/requirements_captioner.txt
# --ignore-installed: base image ships some Debian Python pkgs (e.g. psutil)
# without pip RECORD metadata, which otherwise breaks upgrades.
RUN python3 -m pip install --no-cache-dir --break-system-packages --ignore-installed \
      -r /tmp/requirements_captioner.txt \
    && python3 -m pip install --no-cache-dir --break-system-packages --ignore-installed gdown \
    && rm -f /tmp/requirements_captioner.txt

# Hugging Face cache lives under the docker user home (compose can bind-mount it).
ENV HF_HOME=/home/docker/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/docker/.cache/huggingface/hub \
    HUGGINGFACE_HUB_CACHE=/home/docker/.cache/huggingface/hub \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    CAPTIONER_OFFLINE=1
RUN mkdir -p /home/docker/.cache/huggingface && chown -R docker:docker /home/docker/.cache

USER docker

# Copy and compile TARE first (takes 90s, but is heavily cached and rarely changes)
COPY src/tare_planner /home/docker/ai_module/src/tare_planner
WORKDIR /home/docker/ai_module
RUN /bin/bash -c "\
    source /opt/ros/jazzy/setup.bash && \
    colcon build \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      --packages-select tare_planner"

# ---------------------------------------------------------------------------
# Perception deps for sam_mapper. Above the source COPYs so they stay cached across
# code edits, and split into small layers so a network timeout loses only that layer.
# Cache mounts (not --no-cache-dir) let a retry resume; re-run `just up` if one fails.
USER root
ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Debian-packaged Python modules have no RECORD file, so pip cannot uninstall them
# ("Cannot uninstall psutil 5.9.8, RECORD file not found") and any dependency wanting a
# different version kills the layer. --ignore-installed writes pip-managed copies with
# proper RECORD files into dist-packages, shadowing the dpkg ones.
# Excluded on purpose: numpy (pinned below — rclpy/cv_bridge use its ABI) and
# setuptools (colcon's ament_python builds break if it moves).
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --ignore-installed \
      psutil PyYAML cryptography \
      requests urllib3 certifi idna charset-normalizer \
      packaging six filelock attrs typing_extensions \
      pillow jinja2 markupsafe

# Pin numpy to the base image's version for every pip install below. rclpy, cv_bridge
# and sensor_msgs_py are C extensions built against its ABI: a silent 1.x -> 2.x bump
# does not fail the build, it segfaults at runtime. A resolution error is the safe
# failure. (`|| true` -> empty constraints if the base image has no numpy.)
RUN { python3 -c "import numpy; print('numpy==' + numpy.__version__)" || true; } \
      > /etc/pip-constraints.txt \
 && chmod 0644 /etc/pip-constraints.txt \
 && echo "pip constraints: $(cat /etc/pip-constraints.txt)"
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

# Only if missing: the base image's torch is built against its own CUDA, and replacing
# it with a generic PyPI wheel can break GPU support for the rest of the stack.
RUN --mount=type=cache,target=/root/.cache/pip \
    if python3 -c "import torch, sys; v=tuple(int(x) for x in torch.__version__.split('.')[:2]); \
                   print('found torch', torch.__version__); sys.exit(0 if v >= (2,4) else 1)"; then \
      echo "torch already present, leaving it alone"; \
    else \
      echo "installing torch" && python3 -m pip install "torch>=2.7" torchvision; \
    fi

# Unpinned: guessing which release carries SAM 3 would either be wrong or hold back a
# newer wheel; the import check at the bottom is what proves Sam3VideoModel exists.
# The `|| --ignore-installed` retry here and below catches any apt-shipped transitive
# dep not pre-empted above, instead of failing the build.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade transformers \
    || python3 -m pip install --ignore-installed --upgrade transformers

# OPTIONAL extras — `|| echo` because neither affects correctness and neither is worth
# failing a build over. Without `kernels`, SAM 3 skips NMS/hole-filling/sprinkle removal
# (coarser masks); without `flash-attn`, the backend falls back to sdpa (slower).
# Pinned: transformers' is_kernels_available() requires 0.15.2 <= version < 0.16.0
# (strict upper bound) — unpinned installs grab 0.16.0 and silently fail the check.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install "kernels>=0.15.2,<0.16.0" \
    || echo "OPTIONAL: kernels unavailable — SAM 3 mask post-processing (NMS, hole \
filling, sprinkle removal) will be skipped. Not fatal."
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-build-isolation --no-deps flash-attn \
    || echo "OPTIONAL: flash-attn unavailable — backend falls back to sdpa. Not fatal."

# semantic_mapping's 3D stage: open3d is the big one (~400 MB), hence its own layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install open3d \
    || python3 -m pip install --ignore-installed open3d
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install scipy scikit-learn line_profiler \
    || python3 -m pip install --ignore-installed scipy scikit-learn line_profiler

# cv2 normally arrives with cv_bridge in the ROS base image; only install if missing.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -c "import cv2; print('found cv2', cv2.__version__)" \
    || python3 -m pip install opencv-python-headless

# spacy must genuinely work, not merely import: utils.py:60 calls spacy.load() at module
# scope, reached via semantic_map.py:16, so ObjMapper fails without it. SAM 3 does not
# use it. Retry covers setuptools/click/pydantic possibly being apt-installed.
RUN --mount=type=cache,target=/root/.cache/pip \
    { python3 -m pip install spacy || python3 -m pip install --ignore-installed spacy; } \
 && python3 -m spacy download en_core_web_sm
USER docker

# semantic_mapper is a plain pip package (setup.cfg only, no package.xml/setup.py), so
# it is pip-installed rather than colcon-built. --no-deps: nothing in external/ is
# installed — Grounded-SAM-2 is unused, and ByteTrack is covered by the shim below.
COPY src/semantic_mapper /home/docker/ai_module/src/semantic_mapper
USER root
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-deps -e /home/docker/ai_module/src/semantic_mapper
USER docker

# Copy and compile active dev packages (takes <5s and updates instantly on edit)
COPY src/dummy_vlm /home/docker/ai_module/src/dummy_vlm
COPY src/smart_vlm /home/docker/ai_module/src/smart_vlm
COPY src/sam_mapper /home/docker/ai_module/src/sam_mapper
COPY src/captioner /home/docker/ai_module/src/captioner
RUN /bin/bash -c "\
    source /opt/ros/jazzy/setup.bash && \
    colcon build \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      --packages-select dummy_vlm smart_vlm sam_mapper captioner"

# Ensure the updated install is sourced in every new shell
RUN echo "source /home/docker/ai_module/install/setup.bash" >> /home/docker/.bashrc \
 && echo 'export PATH=/home/docker/ai_module/install/captioner/lib/captioner:$PATH' >> /home/docker/.bashrc

# Fail the build, not the first run, if the import chain is broken. sam_mapper owns its full
# 3D-mapping stack (docs/M2_perception.md 3.6-split — object_mapper.py, cloud_image_fusion.py,
# single_object.py) and no longer depends on semantic_mapping, so no compat/ import shims are
# needed either. Needs ROS sourced (object_mapper.py imports rclpy transitively via ros_markers).
RUN /bin/bash -c "\
    source /home/docker/ai_module/install/setup.bash && \
    python3 -c \"\
import transformers; \
from transformers import Sam3VideoModel, Sam3VideoProcessor; \
from sam_mapper.object_mapper import ObjMapper; \
from sam_mapper.cloud_image_fusion import CloudImageFusion; \
from sam_mapper.detections import PromptTable; \
from sam_mapper.sam3_backend import Sam3Backend; \
print('import check OK: transformers', transformers.__version__)\""

# SAM 3 weights are gated: run `hf auth login` once inside the container
# (or set HF_TOKEN via docker/.env). Override HF_HUB_OFFLINE=0 for the download.
