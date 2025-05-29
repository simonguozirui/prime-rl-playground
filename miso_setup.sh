#!/bin/bash

# Set up environment variables
export VIRTUAL_ENV=/scr-ssd/simonguo/venv-prime-rl
export UV_CACHE_DIR=/scr-ssd/simonguo/uv-cache

# Create and activate virtual environment
# python -m venv "$VIRTUAL_ENV"
source "$VIRTUAL_ENV/bin/activate"

# Install dependencies
uv sync --active

# Add HuggingFace cache directory to .bashrc
export HF_HOME=/scr-ssd/simonguo/huggingface-cache