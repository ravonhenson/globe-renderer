#!/usr/bin/env bash
set -euo pipefail


cmake -S . -B build
cmake --build build --config Debug
exec ./build/Debug/vulkan_triangle.exe