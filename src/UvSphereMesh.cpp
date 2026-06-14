#include "UvSphereMesh.h"

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <cmath>
#include <utility>

namespace {

constexpr float kSphereRadius = 1.0f;
constexpr uint32_t kStacks = 90;   // latitude divisions, pole to pole
constexpr uint32_t kSlices = 180;  // longitude divisions, seam to seam

} // namespace

void buildUvSphereMesh(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) {
    vertices.clear();
    indices.clear();

    vertices.reserve(static_cast<size_t>(kStacks + 1) * (kSlices + 1));
    for (uint32_t i = 0; i <= kStacks; ++i) {
        const float v = static_cast<float>(i) / static_cast<float>(kStacks);
        const float colatitude = v * glm::pi<float>();
        const float y = std::cos(colatitude);
        const float ringRadius = std::sin(colatitude);

        for (uint32_t j = 0; j <= kSlices; ++j) {
            const float texU = static_cast<float>(j) / static_cast<float>(kSlices);
            const float phi = (1.0f - texU) * glm::two_pi<float>();

            const glm::vec3 direction{ringRadius * std::cos(phi), y, ringRadius * std::sin(phi)};

            Vertex vertex{};
            vertex.position = direction * kSphereRadius;
            vertex.normal = direction;
            vertex.texCoord = {texU, v};
            vertices.push_back(vertex);
        }
    }

    const uint32_t rowStride = kSlices + 1;
    for (uint32_t i = 0; i < kStacks; ++i) {
        for (uint32_t j = 0; j < kSlices; ++j) {
            const uint32_t topLeft = i * rowStride + j;
            const uint32_t topRight = topLeft + 1;
            const uint32_t bottomLeft = topLeft + rowStride;
            const uint32_t bottomRight = bottomLeft + 1;

            indices.push_back(topLeft);
            indices.push_back(bottomLeft);
            indices.push_back(topRight);

            indices.push_back(topRight);
            indices.push_back(bottomLeft);
            indices.push_back(bottomRight);
        }
    }

    // Fix winding so every face is wound clockwise as seen from outside the
    // sphere - this matches the cull/front-face configuration used by the
    // graphics pipeline.
    for (size_t t = 0; t < indices.size(); t += 3) {
        const glm::vec3& a = vertices[indices[t]].position;
        const glm::vec3& b = vertices[indices[t + 1]].position;
        const glm::vec3& c = vertices[indices[t + 2]].position;
        const glm::vec3 faceNormal = glm::cross(b - a, c - a);
        if (glm::dot(faceNormal, a + b + c) > 0.0f) {
            std::swap(indices[t + 1], indices[t + 2]);
        }
    }
}
