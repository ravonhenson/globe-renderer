# Globe Renderer

Minimal C++/CMake Vulkan app that renders a rotating, textured 3D globe using
the Natural Earth raster basemap.

## Prerequisites

- Vulkan SDK with `glslc` available
- CMake 3.20+
- A C++17 compiler
- `assets/natural-earth-raster.tif` (uncompressed 8-bit RGB TIFF) present in
  the project's `assets/` directory
- `assets/ETOPO1_Ice_g_geotiff.tif`

## Build

```bash
cmake -S . -B build
cmake --build build
```

GLFW and GLM are fetched automatically via CMake's `FetchContent`.

## Run

Run the generated `vulkan_triangle` executable from the build directory. The
working directory should be (or be near) the project root so that
`assets/natural-earth-raster.tif` and the compiled `shaders/` directory can be
located.

On launch, the app loads the Natural Earth raster (downsampling it on the fly
if it exceeds the GPU's maximum texture dimensions), maps it onto an
icosphere, and renders it with simple directional lighting while slowly
rotating.
