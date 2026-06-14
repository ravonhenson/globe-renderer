#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

// A decoded RGBA8 image, ready to be uploaded to a Vulkan texture.
struct RgbaImage {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> pixels; // width * height * 4 bytes, row-major, top row first
};

// Loads an uncompressed, 8-bit-per-channel RGB (or RGBA) TIFF file.
//
// If the source image is larger than maxDimension in either axis, it is
// box-downsampled (by an integer factor) while streaming from disk so that
// both output dimensions are <= maxDimension. This keeps very large source
// rasters (e.g. whole-earth basemaps) from exceeding GPU texture limits or
// requiring the entire file to be held in memory at once.
RgbaImage loadTiffRgba(const std::filesystem::path& path, uint32_t maxDimension);
