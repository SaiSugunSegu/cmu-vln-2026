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
# Pin numpy BEFORE captioner deps: an unpinned install can pull numpy 2.x, which
# breaks cv_bridge (built against Debian 1.26) at runtime (segfault / ImportError).
# rclpy, cv_bridge and sensor_msgs_py are C extensions built against the base
# image's numpy ABI, so a silent 1.x -> 2.x bump does not fail the build, it
# segfaults at runtime. A pip resolution error is the safe failure.
# This generated file is the single source of truth for the numpy version — do
# NOT also pin numpy in requirements_captioner.txt, or a base-image bump turns
# into an unresolvable conflict. (`|| true` -> empty constraints if the base
# image has no numpy at all.)
RUN { python3 -c "import numpy; print('numpy==' + numpy.__version__)" || true; } \
      > /etc/pip-constraints.txt \
 && chmod 0644 /etc/pip-constraints.txt \
 && echo "pip constraints: $(cat /etc/pip-constraints.txt)"
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt
# Applies to every pip layer below, this one included — the defaults are too tight
# for the multi-GB torch wheel that requirements_captioner.txt pulls next.
# PIP_BREAK_SYSTEM_PACKAGES replaces a --break-system-packages flag per invocation.
ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --ignore-installed: base image ships some Debian Python pkgs (e.g. psutil)
# without pip RECORD metadata, which otherwise breaks upgrades.
# Cache mount rather than --no-cache-dir so a network failure part-way through
# torch resumes on the next attempt instead of re-downloading from zero. The
# cache lives outside the layer either way, so the image is no larger.
# gdown is not a captioner dep: bag_fetch / scene_fetch use it to pull Unity scenes
# and recorded bags from the organizers' public Drive folder.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --ignore-installed -r /tmp/requirements_captioner.txt \
    && python3 -m pip install --ignore-installed gdown \
    && rm -f /tmp/requirements_captioner.txt

# Hugging Face cache lives under the docker user home. just up bakes weights here.
# HF_HUB_OFFLINE deliberately NOT set: this deployment always has connectivity, and the
# cv-utils kernel downloads on first use (its absence silently disables mask NMS).
# HUGGINGFACE_HUB_CACHE is not set: HF derives $HF_HOME/hub by default.
# TRANSFORMERS_CACHE / TRANSFORMERS_OFFLINE are deprecated aliases; don't add them.
ENV HF_HOME=/home/docker/.cache/huggingface
RUN mkdir -p /home/docker/.cache/huggingface /home/docker/crops \
 && chown -R docker:docker /home/docker/.cache /home/docker/crops

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

# PIP_CONSTRAINT already set above (base-image numpy). Keep it for layers below.

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

# `kernels` delivers kernels-community/cv-utils (no pip package exists), which SAM 3 uses
# for two jobs: generic_nms (mask NMS — skipped entirely without it, making det_nms_thresh
# dead) and cc_2d (hole filling, the only half gated by fill_hole_area). Both fail silently.
# A matching prebuilt variant is also needed — see the torch pin in requirements_captioner.txt.
#
# The version window is DERIVED from the installed transformers, which is unpinned: a
# hardcoded pin rots on the next rebuild, and did (5.14.1 wanted >=0.15.2,<0.16.0; 5.15.0
# wants >=0.16.0,<0.17.0, and the mismatch silently disabled every kernel).
# No `|| echo`: swallowing this failure is what let it run degraded for months.
RUN --mount=type=cache,target=/root/.cache/pip \
    spec="$(python3 -c 'from transformers.utils.import_utils import KERNELS_MIN_VERSION, KERNELS_MAX_VERSION; print(f"kernels>={KERNELS_MIN_VERSION},<{KERNELS_MAX_VERSION}")')" \
    && echo "transformers requires: $spec" \
    && python3 -m pip install "$spec" \
    && python3 -c "\
from transformers.utils import is_kernels_available; \
import kernels; \
assert is_kernels_available(), f'kernels {kernels.__version__} installed but transformers still rejects it'; \
print('kernels OK:', kernels.__version__)"
# NOT installed, deliberately, and each for its own reason:
#   flash-attn (pip)  harmful: with it installed transformers runs real FA2, which on SAM 3
#                     measures 0.0 objects/frame vs sdpa's 2.0 (probe: --attn <impl>).
#   flash-attn-3      Hopper-only (sm_90); both deploy targets are Ada sm_89.
#   cc_torch          named only in a transformers DOCSTRING inherited from the native repo;
#                     the code calls cv_utils_kernel.cc_2d. Installing it does nothing.
#   ninja             only mattered for the source builds we no longer do.
#                     (einops IS needed — but by the native sam3 package below, not by
#                     transformers' sam3* modules.)

# Native SAM 3.1 (Object Multiplex). transformers has NO multiplex support, so this is the
# only way to get it — the sam3.1_multiplex.pt checkpoint loads here directly, no conversion.
# Pinned to a sha: the integration drives internals (_run_single_frame_inference and the
# per-frame session lists) that carry no API guarantee; tests/test_sam31_contract.py fails
# loudly if a bump moves them.
# --no-deps is load-bearing: sam3 pins ftfy==6.1.1 and would downgrade ours (open_clip needs
# it), and an unconstrained resolve could move torch or numpy — numpy 1.26's ABI is what
# cv_bridge/rclpy are built against.
#
# So its deps are named explicitly. This list is what the import chain from
# build_sam3_multiplex_video_predictor actually reaches, not sam3's pyproject:
#   einops        sam3/sam/rope.py:15 — top-level, and NOT declared in sam3's pyproject
#                 (it sits under the `notebooks` extra), so --no-deps is not to blame
#   pycocotools   sam3/train/masks_ops.py:8, pulled top-level by sam3_multiplex_detector_utils
#                 and sam3_video_base
#   iopath        declared dep, used by model_builder
# Deliberately NOT installed, having checked each: hydra (imported inside a function, for the
# config-driven build path we bypass), skimage (only visualization_utils, off the model path),
# and cc_torch / decord / torchcodec / xformers / torch_generic_nms / flash_attn_interface
# (all deferred or try-guarded).
ARG SAM31_SHA=96914d2425f90a64f45ca977c2b5165418099543
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install einops iopath pycocotools \
    && python3 -m pip install --no-deps \
       "git+https://github.com/facebookresearch/sam3.git@${SAM31_SHA}" \
    && python3 -c "\
import ftfy, numpy, torch; \
from sam3.model_builder import build_sam3_multiplex_video_predictor; \
print(f'sam3 OK — ftfy {ftfy.__version__}, numpy {numpy.__version__}, torch {torch.__version__}')"

# semantic_mapping's 3D stage: open3d is the big one (~400 MB), hence its own layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install open3d \
    || python3 -m pip install --ignore-installed open3d
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install "scipy==1.13.1" scikit-learn line_profiler \
    || python3 -m pip install --ignore-installed "scipy==1.13.1" scikit-learn line_profiler

# cv2 normally arrives with cv_bridge in the ROS base image; only install if missing.
# Pin <5: opencv 5.x requires numpy>=2, which breaks cv_bridge.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -c "import cv2; print('found cv2', cv2.__version__)" \
    || python3 -m pip install "opencv-python-headless==4.10.0.84"

# spacy must genuinely work, not merely import: utils.py:60 calls spacy.load() at module
# scope, reached via semantic_map.py:16, so ObjMapper fails without it. SAM 3 does not
# use it. Retry covers setuptools/click/pydantic possibly being apt-installed.
RUN --mount=type=cache,target=/root/.cache/pip \
    { python3 -m pip install spacy || python3 -m pip install --ignore-installed spacy; } \
 && python3 -m spacy download en_core_web_sm
USER docker

# Bake SAM 3 into HF_HOME BEFORE the package COPYs so a code edit does not
# re-download ~6.5 GB. Qwen3-VL is the host HF cache; `just vqa-up` bind-mounts it.
# fetch_weights.py is stdlib + huggingface_hub + transformers (already installed
# above); it does not need the captioner colcon install. Cache mount lets a retry
# of this layer skip files already pulled. Empty HF_TOKEN skips the download;
# runtime from_pretrained still reads ENV.
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}
COPY --chown=docker:docker src/captioner/captioner/tools/fetch_weights.py /tmp/fetch_weights.py
RUN --mount=type=cache,target=/tmp/hf-build-cache,uid=1001,gid=1001 \
    if [ -n "$HF_TOKEN" ]; then \
      HF_HOME=/tmp/hf-build-cache python3 /tmp/fetch_weights.py sam3 && \
      mkdir -p /home/docker/.cache/huggingface && \
      cp -a /tmp/hf-build-cache/. /home/docker/.cache/huggingface/; \
    else \
      echo "HF_TOKEN empty — skipping weight bake"; \
    fi

# Copy and compile active dev packages (takes <5s and updates instantly on edit)
COPY src/smart_vlm /home/docker/ai_module/src/smart_vlm
COPY src/sam_mapper /home/docker/ai_module/src/sam_mapper
COPY src/captioner /home/docker/ai_module/src/captioner
COPY src/dummy_vlm /home/docker/ai_module/src/dummy_vlm
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
from captioner.paths import ALLOWED_ROOTS; \
from captioner.text_utils import extract_integer; \
from captioner.vlm_backends.schemas import CountAnswer, RoutePlan, TargetList; \
from captioner.vlm_backends.openai_backend import OpenAIBackend; \
from smart_vlm.numerical_utils import EXTRACT_SYSTEM; \
from smart_vlm.cat2_utils import EXTRACT_SYSTEM as CAT2_EXTRACT; \
from smart_vlm.instruction_reasoner import ROUTE_SYSTEM; \
import openai; \
print('import check OK: transformers', transformers.__version__)\""
