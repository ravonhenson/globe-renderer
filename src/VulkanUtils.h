#pragma once

#include <vulkan/vulkan.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

inline void checkVk(VkResult result, const char* message) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(message);
    }
}

inline std::vector<char> readFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + path.string());
    }

    const std::streamsize size = file.tellg();
    std::vector<char> buffer(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), size);
    return buffer;
}
