FROM zhangjicmu/ubuntu24_ros:cmu_vla_challenge_simulation
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-domain-bridge xvfb \
    && rm -rf /var/lib/apt/lists/*

# VirtualGL lets Unity render on the NVIDIA GPU via EGL and blit the result to
# Xvfb, which has no GL acceleration of its own. mesa-utils provides glxinfo
# for checking which renderer you actually got.
ARG VIRTUALGL_VERSION=3.1.4
# sha256 of virtualgl_${VIRTUALGL_VERSION}_amd64.deb from the official GitHub
# release. Verified before install so a changed/MITM'd asset fails the build.
# Update this hash whenever VIRTUALGL_VERSION changes.
ARG VIRTUALGL_SHA256=02edc6b599571c385389af1a006f07a70c298e1d97c580a9bfd4b39d835c51e6
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl mesa-utils \
    && curl -fsSL -o /tmp/virtualgl.deb \
      "https://github.com/VirtualGL/virtualgl/releases/download/${VIRTUALGL_VERSION}/virtualgl_${VIRTUALGL_VERSION}_amd64.deb" \
    && echo "${VIRTUALGL_SHA256}  /tmp/virtualgl.deb" | sha256sum -c - \
    && apt-get install -y --no-install-recommends /tmp/virtualgl.deb \
    && rm -f /tmp/virtualgl.deb \
    && rm -rf /var/lib/apt/lists/*

# The container toolkit injects libEGL_nvidia.so.0 but not this vendor config,
# so libglvnd would only ever find Mesa and Unity would fall back to llvmpipe.
RUN printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
      > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Challenge-mode entry points, alongside the stack's own system_simulation.sh
COPY --chmod=755 challenge_simulation.sh system_simulation_noviz.sh \
      /home/docker/autonomy_stack_mecanum_wheel_platform/
COPY challenge_topics_bridge.yaml /home/docker/autonomy_stack_mecanum_wheel_platform/

USER docker
