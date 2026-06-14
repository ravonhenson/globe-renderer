#include "IcosphereMesh.h"

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <utility>

namespace {

constexpr float kSphereRadius = 1.0f;
constexpr uint32_t kSubdivisions = 6;
constexpr float kPoleEpsilonSq = 1e-12f;

using Face = std::array<uint32_t, 3>;

// Builds the base icosahedron: a north-pole vertex, a ring of five
// vertices just below it, a ring of five vertices (offset by half a step)
// just above the south pole, and a south-pole vertex. Keeping a single
// vertex exactly at each pole keeps pole detection simple: only vertex 0
// and vertex 11 ever have x == z == 0, and subdivision never creates new
// vertices there.
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

struct EdgeKey {
    uint32_t a;
    uint32_t b;

    bool operator==(const EdgeKey& other) const { return a == other.a && b == other.b; }
};

struct EdgeKeyHash {
    size_t operator()(const EdgeKey& key) const {
        return std::hash<uint64_t>{}((static_cast<uint64_t>(key.a) << 32) | key.b);
    }
};

// Subdivides every face into 4 by inserting sphere-projected edge
// midpoints, reusing a single midpoint vertex for both faces that share an
// edge.
void subdivide(std::vector<glm::vec3>& positions, std::vector<Face>& faces) {
    std::unordered_map<EdgeKey, uint32_t, EdgeKeyHash> midpointCache;

    const auto midpoint = [&](uint32_t a, uint32_t b) -> uint32_t {
        const EdgeKey key{std::min(a, b), std::max(a, b)};
        const auto it = midpointCache.find(key);
        if (it != midpointCache.end()) {
            return it->second;
        }
        const glm::vec3 mid = glm::normalize(positions[a] + positions[b]);
        const uint32_t index = static_cast<uint32_t>(positions.size());
        positions.push_back(mid);
        midpointCache.emplace(key, index);
        return index;
    };

    std::vector<Face> nextFaces;
    nextFaces.reserve(faces.size() * 4);
    for (const Face& face : faces) {
        const uint32_t a = face[0];
        const uint32_t b = face[1];
        const uint32_t c = face[2];
        const uint32_t ab = midpoint(a, b);
        const uint32_t bc = midpoint(b, c);
        const uint32_t ca = midpoint(c, a);

        nextFaces.push_back({a, ab, ca});
        nextFaces.push_back({b, bc, ab});
        nextFaces.push_back({c, ca, bc});
        nextFaces.push_back({ab, bc, ca});
    }
    faces = std::move(nextFaces);
}

// Per-vertex texture coordinates derived from the vertex's direction,
// matching the convention used by the UV sphere it replaces:
//   texV = colatitude / pi          (0 at the north pole, 1 at the south)
//   texU = 1 - atan2(z, x) / (2*pi)  (wrapped into [0, 1))
// Pole vertices (x == z == 0) have undefined longitude; texU is resolved
// per-triangle instead (see buildIcosphereMesh).
struct VertexUV {
    float texU;
    float texV;
    bool isPole;
};

std::vector<VertexUV> computeVertexUVs(const std::vector<glm::vec3>& positions) {
    std::vector<VertexUV> uvs(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        const glm::vec3& p = positions[i];
        uvs[i].texV = std::acos(glm::clamp(p.y, -1.0f, 1.0f)) / glm::pi<float>();

        if (p.x * p.x + p.z * p.z < kPoleEpsilonSq) {
            uvs[i].isPole = true;
            uvs[i].texU = 0.0f;
            continue;
        }

        uvs[i].isPole = false;
        float u = 1.0f - std::atan2(p.z, p.x) / glm::two_pi<float>();
        if (u >= 1.0f) {
            u -= 1.0f;
        }
        uvs[i].texU = u;
    }
    return uvs;
}

} // namespace

void buildIcosphereMesh(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) {
    std::vector<glm::vec3> positions;
    std::vector<Face> faces;
    buildBaseIcosahedron(positions, faces);

    for (uint32_t level = 0; level < kSubdivisions; ++level) {
        subdivide(positions, faces);
    }

    const std::vector<VertexUV> uvs = computeVertexUVs(positions);

    vertices.clear();
    indices.clear();
    vertices.reserve(positions.size() + 512);
    indices.reserve(faces.size() * 3);

    // Vertices on the antimeridian seam or at a pole need more than one
    // texU for a single position - one per side of the seam, or one per
    // longitude wedge meeting at a pole. Each base vertex maps to a small
    // set of (quantized texU -> output vertex index) variants, created on
    // demand and shared by every triangle that needs the same texU.
    std::vector<std::unordered_map<int32_t, uint32_t>> variants(positions.size());

    const auto quantize = [](float u) -> int32_t {
        return static_cast<int32_t>(std::lround(u * 1'000'000.0f));
    };

    const auto getOrCreateVertex = [&](uint32_t base, float texU) -> uint32_t {
        std::unordered_map<int32_t, uint32_t>& cache = variants[base];
        const int32_t key = quantize(texU);
        const auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second;
        }

        Vertex vertex{};
        vertex.position = positions[base] * kSphereRadius;
        vertex.normal = positions[base];
        vertex.texCoord = {texU, uvs[base].texV};

        const uint32_t index = static_cast<uint32_t>(vertices.size());
        vertices.push_back(vertex);
        cache.emplace(key, index);
        return index;
    };

    for (const Face& face : faces) {
        std::array<float, 3> texU{};
        std::array<bool, 3> isPole{};
        for (int k = 0; k < 3; ++k) {
            texU[k] = uvs[face[k]].texU;
            isPole[k] = uvs[face[k]].isPole;
        }

        // Unwrap texU across the antimeridian seam so a triangle that
        // straddles texU = 0/1 interpolates continuously (the REPEAT
        // sampler wraps values like 1.02 back to 0.02 at lookup time).
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

        // A pole vertex's texU is undefined; use the average of this
        // triangle's other two (now seam-unwrapped) corners so the wedge
        // of texture between them extends smoothly to the pole.
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
                texU[k] = sum / static_cast<float>(count);
            }
        }

        for (int k = 0; k < 3; ++k) {
            indices.push_back(getOrCreateVertex(face[k], texU[k]));
        }
    }

    // Fix winding so every face is wound clockwise as seen from outside
    // the sphere - this matches the cull/front-face configuration used by
    // the graphics pipeline.
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
