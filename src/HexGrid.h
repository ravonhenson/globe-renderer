#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

// Two-attribute vertex used exclusively by the hex grid pipeline.
struct HexVertex {
    glm::vec3 pos;
    glm::vec3 color;

    static VkVertexInputBindingDescription bindingDescription() {
        VkVertexInputBindingDescription desc{};
        desc.binding   = 0;
        desc.stride    = sizeof(HexVertex);
        desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return desc;
    }

    static std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> descs{};
        descs[0].binding  = 0;
        descs[0].location = 0;
        descs[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
        descs[0].offset   = offsetof(HexVertex, pos);
        descs[1].binding  = 0;
        descs[1].location = 1;
        descs[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
        descs[1].offset   = offsetof(HexVertex, color);
        return descs;
    }
};

namespace HexGrid {

// Fills outVertices with a LINE_LIST wireframe that tiles the entire unit
// sphere using the dual of a subdivided icosphere (12 pentagons +
// (10·4^N − 10) hexagons). subdivisions controls resolution: each step
// quadruples the face count. N=6 gives ~40 k cells with circumradius ≈ 0.016.
void generateSphericalWireframe(int subdivisions,
                                std::vector<HexVertex>& outVertices);

// Fills outVertices with a LINE_LIST of outward normals, one per cell,
// originating at each cell centre on the unit sphere and extending radially
// outward by `length` (in unit-sphere units). Uses the same dual construction
// as generateSphericalWireframe so the normals are centred on the hex cells.
void generateNormalLines(int subdivisions, float length,
                         std::vector<HexVertex>& outVertices);

} // namespace HexGrid
