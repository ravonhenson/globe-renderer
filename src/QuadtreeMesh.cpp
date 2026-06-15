#include "QuadtreeMesh.h"

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace QuadtreeMesh {

namespace {

constexpr float kPoleEpsilonSq = 1e-12f;

// Per-vertex texture coordinates derived from the vertex's direction, same
// convention as IcosphereMesh: texV = colatitude / pi, texU = wrapped
// longitude. Pole points (x == z == 0) have undefined longitude.
struct VertexUV {
    float texU;
    float texV;
    bool isPole;
};

VertexUV computeUV(const glm::vec3& p) {
    VertexUV uv{};
    uv.texV = std::acos(glm::clamp(p.y, -1.0f, 1.0f)) / glm::pi<float>();

    if (p.x * p.x + p.z * p.z < kPoleEpsilonSq) {
        uv.isPole = true;
        uv.texU = 0.0f;
        return uv;
    }

    uv.isPole = false;
    float u = 1.0f - std::atan2(p.z, p.x) / glm::two_pi<float>();
    if (u >= 1.0f) {
        u -= 1.0f;
    }
    uv.texU = u;
    return uv;
}

using Face = std::array<uint32_t, 3>;

// Same construction as IcosphereMesh::buildBaseIcosahedron: a north-pole
// vertex, two pentagonal rings, and a south-pole vertex, forming the 20
// quadtree roots.
void buildBaseIcosahedron(std::vector<glm::vec3>& positions, std::vector<Face>& faces) {
    positions.clear();
    faces.clear();
    positions.reserve(12);

    const float ringY = 1.0f / std::sqrt(5.0f);
    const float ringRadius = 2.0f / std::sqrt(5.0f);

    positions.push_back({0.0f, 1.0f, 0.0f}); // 0: north pole
    for (uint32_t i = 0; i < 5; ++i) {
        const float angle = glm::two_pi<float>() * static_cast<float>(i) / 5.0f;
        positions.push_back({ringRadius * std::cos(angle), ringY, ringRadius * std::sin(angle)});
    }
    for (uint32_t i = 0; i < 5; ++i) {
        const float angle = glm::two_pi<float>() * (static_cast<float>(i) + 0.5f) / 5.0f;
        positions.push_back({ringRadius * std::cos(angle), -ringY, ringRadius * std::sin(angle)});
    }
    positions.push_back({0.0f, -1.0f, 0.0f}); // 11: south pole

    const auto upper = [](uint32_t i) { return 1 + (i % 5); };
    const auto lower = [](uint32_t i) { return 6 + (i % 5); };

    faces.reserve(20);
    for (uint32_t i = 0; i < 5; ++i) {
        faces.push_back({0, upper(i), upper(i + 1)});            // north cap
        faces.push_back({11, lower(i + 1), lower(i)});           // south cap
        faces.push_back({upper(i), upper(i + 1), lower(i)});     // middle band, "down" triangle
        faces.push_back({lower(i), lower(i + 1), upper(i + 1)}); // middle band, "up" triangle
    }
}

// Splits a quadtree node into 4 children when the camera is close enough
// relative to the node's size, down to kMaxDepth, appending leaves to
// `leaves`. kMaxLeaves is a safety cap on total output size.
void selectRecursive(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, uint32_t depth,
                      const glm::vec3& cameraPos, std::vector<Patch>& leaves) {
    if (leaves.size() >= kMaxLeaves) {
        leaves.push_back({a, b, c, depth});
        return;
    }

    const glm::vec3 center = glm::normalize(a + b + c);
    const float edgeLength = glm::length(b - a);
    const float distance = glm::length(cameraPos - center);

    if (depth < kMaxDepth && distance < edgeLength * kLodDistanceFactor) {
        const glm::vec3 ab = glm::normalize(a + b);
        const glm::vec3 bc = glm::normalize(b + c);
        const glm::vec3 ca = glm::normalize(c + a);
        selectRecursive(a, ab, ca, depth + 1, cameraPos, leaves);
        selectRecursive(b, bc, ab, depth + 1, cameraPos, leaves);
        selectRecursive(c, ca, bc, depth + 1, cameraPos, leaves);
        selectRecursive(ab, bc, ca, depth + 1, cameraPos, leaves);
    } else {
        leaves.push_back({a, b, c, depth});
    }
}

// Index of grid point (i, j) (i + j <= kGridSize) within a flattened
// triangular grid, row-major by i.
constexpr uint32_t gridIndex(uint32_t i, uint32_t j) {
    return i * (2 * kGridSize + 3 - i) / 2 + j;
}

// Tessellates one leaf patch into a kGridSize x kGridSize barycentric grid
// plus a skirt, appending vertices/indices for it to the output arrays.
void buildPatchMesh(const Patch& patch, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) {
    std::array<glm::vec3, kGridPointCount> gridPos{};
    std::array<VertexUV, kGridPointCount> gridUV{};

    for (uint32_t i = 0; i <= kGridSize; ++i) {
        for (uint32_t j = 0; j + i <= kGridSize; ++j) {
            const uint32_t k = kGridSize - i - j;
            const glm::vec3 p = glm::normalize(
                patch.a * (static_cast<float>(k) / static_cast<float>(kGridSize)) +
                patch.b * (static_cast<float>(i) / static_cast<float>(kGridSize)) +
                patch.c * (static_cast<float>(j) / static_cast<float>(kGridSize)));
            const uint32_t gi = gridIndex(i, j);
            gridPos[gi] = p;
            gridUV[gi] = computeUV(p);
        }
    }

    // Interior grid vertices on the antimeridian seam or at a pole need
    // more than one texU for a single position - one per side of the seam,
    // or one per longitude wedge meeting at a pole. Each grid point maps to
    // a small set of (quantized texU -> output vertex index) variants,
    // created on demand and shared by every triangle that needs the same
    // texU (mirrors IcosphereMesh::buildIcosphereMesh). A fixed-size
    // per-point slot list (instead of std::unordered_map) avoids dozens of
    // heap allocations per leaf - this runs for every leaf on every mesh
    // rebuild, so allocation overhead here was a major cost. In practice a
    // grid point needs at most 2-3 distinct texU values.
    constexpr uint8_t kMaxVariantsPerPoint = 4;
    std::array<std::array<int32_t, kMaxVariantsPerPoint>, kGridPointCount> variantKeys{};
    std::array<std::array<uint32_t, kMaxVariantsPerPoint>, kGridPointCount> variantIndices{};
    std::array<uint8_t, kGridPointCount> variantCounts{};

    const auto quantize = [](float u) -> int32_t {
        return static_cast<int32_t>(std::lround(u * 1'000'000.0f));
    };

    const auto getOrCreateVertex = [&](uint32_t gi, float texU) -> uint32_t {
        const int32_t key = quantize(texU);
        uint8_t& count = variantCounts[gi];
        for (uint8_t v = 0; v < count; ++v) {
            if (variantKeys[gi][v] == key) {
                return variantIndices[gi][v];
            }
        }

        Vertex vertex{};
        vertex.position = gridPos[gi];
        vertex.normal = gridPos[gi];
        vertex.texCoord = {texU, gridUV[gi].texV};

        const uint32_t index = static_cast<uint32_t>(vertices.size());
        vertices.push_back(vertex);
        variantKeys[gi][count] = key;
        variantIndices[gi][count] = index;
        ++count;
        return index;
    };

    // Resolves the 3 corners of one small triangle to output vertices,
    // applying the same seam-unwrap / pole-average / winding fix as
    // IcosphereMesh, then appends it.
    const auto emitTriangle = [&](uint32_t gi0, uint32_t gi1, uint32_t gi2) {
        const std::array<uint32_t, 3> gi{gi0, gi1, gi2};
        std::array<float, 3> texU{gridUV[gi0].texU, gridUV[gi1].texU, gridUV[gi2].texU};
        const std::array<bool, 3> isPole{gridUV[gi0].isPole, gridUV[gi1].isPole, gridUV[gi2].isPole};

        float minU = 2.0f;
        float maxU = -1.0f;
        for (int k = 0; k < 3; ++k) {
            if (isPole[k]) continue;
            minU = std::min(minU, texU[k]);
            maxU = std::max(maxU, texU[k]);
        }
        if (maxU - minU > 0.5f) {
            for (int k = 0; k < 3; ++k) {
                if (!isPole[k] && texU[k] < 0.5f) {
                    texU[k] += 1.0f;
                }
            }
        }

        for (int k = 0; k < 3; ++k) {
            if (isPole[k]) {
                float sum = 0.0f;
                int count = 0;
                for (int j = 0; j < 3; ++j) {
                    if (j != k && !isPole[j]) {
                        sum += texU[j];
                        ++count;
                    }
                }
                texU[k] = count > 0 ? sum / static_cast<float>(count) : 0.0f;
            }
        }

        std::array<uint32_t, 3> out{};
        for (int k = 0; k < 3; ++k) {
            out[k] = getOrCreateVertex(gi[k], texU[k]);
        }

        const glm::vec3& a = vertices[out[0]].position;
        const glm::vec3& b = vertices[out[1]].position;
        const glm::vec3& c = vertices[out[2]].position;
        if (glm::dot(glm::cross(b - a, c - a), a + b + c) > 0.0f) {
            std::swap(out[1], out[2]);
        }

        indices.push_back(out[0]);
        indices.push_back(out[1]);
        indices.push_back(out[2]);
    };

    for (uint32_t i = 0; i < kGridSize; ++i) {
        for (uint32_t j = 0; i + j < kGridSize; ++j) {
            const uint32_t p00 = gridIndex(i, j);
            const uint32_t p10 = gridIndex(i + 1, j);
            const uint32_t p01 = gridIndex(i, j + 1);
            emitTriangle(p00, p10, p01);

            if (i + j + 1 < kGridSize) {
                const uint32_t p11 = gridIndex(i + 1, j + 1);
                emitTriangle(p10, p11, p01);
            }
        }
    }

    // Skirt: duplicate each boundary grid point as a "top" vertex (at its
    // natural texU/texV, undoing pole/seam ambiguity is unnecessary here -
    // skirt geometry is a hidden filler) and a "bottom" vertex pulled
    // toward the sphere center, then wall the two rims together. This hides
    // gaps where a neighboring leaf is at a different LOD level.
    constexpr uint32_t kNoVertex = UINT32_MAX;
    std::array<uint32_t, kGridPointCount> skirtTop;
    std::array<uint32_t, kGridPointCount> skirtBottom;
    skirtTop.fill(kNoVertex);
    skirtBottom.fill(kNoVertex);

    const auto getSkirtVertex = [&](uint32_t gi, bool bottom) -> uint32_t {
        uint32_t& slot = bottom ? skirtBottom[gi] : skirtTop[gi];
        if (slot != kNoVertex) {
            return slot;
        }

        Vertex vertex{};
        vertex.position = bottom ? gridPos[gi] * (1.0f - kSkirtDepth) : gridPos[gi];
        vertex.normal = gridPos[gi];
        vertex.texCoord = {gridUV[gi].isPole ? 0.0f : gridUV[gi].texU, gridUV[gi].texV};

        slot = static_cast<uint32_t>(vertices.size());
        vertices.push_back(vertex);
        return slot;
    };

    const auto emitSkirtTriangle = [&](uint32_t v0, uint32_t v1, uint32_t v2) {
        std::array<uint32_t, 3> out{v0, v1, v2};
        const glm::vec3& a = vertices[out[0]].position;
        const glm::vec3& b = vertices[out[1]].position;
        const glm::vec3& c = vertices[out[2]].position;
        if (glm::dot(glm::cross(b - a, c - a), a + b + c) > 0.0f) {
            std::swap(out[1], out[2]);
        }
        indices.push_back(out[0]);
        indices.push_back(out[1]);
        indices.push_back(out[2]);
    };

    std::array<uint32_t, kGridSize + 1> edgeAB{};
    std::array<uint32_t, kGridSize + 1> edgeBC{};
    std::array<uint32_t, kGridSize + 1> edgeCA{};
    for (uint32_t i = 0; i <= kGridSize; ++i) {
        edgeAB[i] = gridIndex(i, 0);
        edgeBC[i] = gridIndex(kGridSize - i, i);
        edgeCA[i] = gridIndex(0, kGridSize - i);
    }

    for (const auto& edge : {edgeAB, edgeBC, edgeCA}) {
        for (uint32_t k = 0; k < kGridSize; ++k) {
            const uint32_t top0 = getSkirtVertex(edge[k], false);
            const uint32_t top1 = getSkirtVertex(edge[k + 1], false);
            const uint32_t bot0 = getSkirtVertex(edge[k], true);
            const uint32_t bot1 = getSkirtVertex(edge[k + 1], true);
            emitSkirtTriangle(top0, top1, bot0);
            emitSkirtTriangle(top1, bot1, bot0);
        }
    }
}

} // namespace

std::vector<Patch> selectLeafPatches(const glm::vec3& cameraObjectPos) {
    std::vector<glm::vec3> positions;
    std::vector<Face> faces;
    buildBaseIcosahedron(positions, faces);

    std::vector<Patch> leaves;
    leaves.reserve(kMaxLeaves);
    for (const Face& face : faces) {
        selectRecursive(positions[face[0]], positions[face[1]], positions[face[2]], 0, cameraObjectPos, leaves);
    }
    return leaves;
}

void generateMesh(const std::vector<Patch>& leaves, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) {
    vertices.clear();
    indices.clear();
    vertices.reserve(leaves.size() * kMaxVerticesPerLeaf);
    indices.reserve(leaves.size() * kMaxIndicesPerLeaf);

    for (const Patch& leaf : leaves) {
        buildPatchMesh(leaf, vertices, indices);
    }
}

} // namespace QuadtreeMesh
