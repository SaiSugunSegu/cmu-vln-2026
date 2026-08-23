#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Build context is ai_module/ so COPY src/ works correctly
cd $SCRIPT_DIR/..
docker build -t iros2026/ai_module:latest -f docker/Dockerfile .
