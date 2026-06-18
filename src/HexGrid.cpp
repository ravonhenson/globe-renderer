#include "HexGrid.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>

namespace {

using IcoVerts = std::vector<glm::vec3>;
using IcoFaces = std::vector<std::array<int, 3>>;

// Rotate the icosahedron so its 12 pentagon vertices (the dual cells with only
// 5 neighbours) fall in ocean tiles rather than on land. H3-style placement.
//
// Resulting pentagon lat/lon (tune with the N key):
//   ~72°N 153°E  Arctic Ocean          ~72°S 153°W  Antarctic
//   ~58°N 120°E  Sea of Okhotsk edge*  ~58°S 120°W  S Pacific
//   ~40°N 290°E  N Atlantic            ~40°S 290°E  S Atlantic
//   ~32°N 218°E  N Pacific             ~32°S  38°E  Indian Ocean
//    ~0°  332°E  S Atlantic             ~0°   88°E  Bay of Bengal
//    ~0°  152°E  W Pacific              ~0°  208°E  C Pacific
//
// (*) The 58°N 120°E cell sits on the coast; bump kIcosaLonOffsetDeg by a few
//     degrees if you want it fully offshore.
constexpr float kIcosaLonOffsetDeg = -60.0f;

glm::mat3 buildIcosaRotation() {
    return glm::mat3(glm::rotate(glm::mat4(1.0f),
                                  glm::radians(kIcosaLonOffsetDeg),
                                  glm::vec3(0.0f, 1.0f, 0.0f)));
}

IcoVerts buildBaseVerts() {
    const glm::mat3 rot = buildIcosaRotation();
    const float phi = (1.0f + std::sqrt(5.0f)) * 0.5f;
    const glm::vec3 raw[] = {
        glm::normalize(glm::vec3(-1,  phi,  0)),
        glm::normalize(glm::vec3( 1,  phi,  0)),
        glm::normalize(glm::vec3(-1, -phi,  0)),
        glm::normalize(glm::vec3( 1, -phi,  0)),
        glm::normalize(glm::vec3( 0, -1,  phi)),
        glm::normalize(glm::vec3( 0,  1,  phi)),
        glm::normalize(glm::vec3( 0, -1, -phi)),
        glm::normalize(glm::vec3( 0,  1, -phi)),
        glm::normalize(glm::vec3( phi,  0, -1)),
        glm::normalize(glm::vec3( phi,  0,  1)),
        glm::normalize(glm::vec3(-phi,  0, -1)),
        glm::normalize(glm::vec3(-phi,  0,  1)),
    };
    IcoVerts result;
    result.reserve(12);
    for (const auto& v : raw)
        result.push_back(rot * v);
    return result;
}

IcoFaces buildBaseFaces() {
    return {{
        {0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11},
        {1,5,9},{5,11,4},{11,10,2},{10,7,6},{7,1,8},
        {3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9},
        {4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1},
    }};
}

int midpointIdx(std::unordered_map<uint64_t, int>& cache,
                IcoVerts& verts, int a, int b) {
    if (a > b) std::swap(a, b);
    const uint64_t key = ((uint64_t)a << 32) | (uint64_t)b;
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    const int idx = (int)verts.size();
    verts.push_back(glm::normalize((verts[a] + verts[b]) * 0.5f));
    cache[key] = idx;
    return idx;
}

void subdivideOnce(IcoVerts& verts, IcoFaces& faces) {
    std::unordered_map<uint64_t, int> cache;
    cache.reserve(faces.size() * 3);
    IcoFaces next;
    next.reserve(faces.size() * 4);
    for (const auto& f : faces) {
        const int m0 = midpointIdx(cache, verts, f[0], f[1]);
        const int m1 = midpointIdx(cache, verts, f[1], f[2]);
        const int m2 = midpointIdx(cache, verts, f[2], f[0]);
        next.push_back({f[0], m0, m2});
        next.push_back({f[1], m1, m0});
        next.push_back({f[2], m2, m1});
        next.push_back({m0,   m1, m2});
    }
    faces = std::move(next);
}

} // namespace

namespace HexGrid {

void generateSphericalWireframe(int subdivisions,
                                std::vector<HexVertex>& out) {
    IcoVerts verts = buildBaseVerts();
    IcoFaces faces = buildBaseFaces();
    for (int i = 0; i < subdivisions; ++i)
        subdivideOnce(verts, faces);

    const int nFaces = (int)faces.size();

    // Dual vertex for each triangular face = centroid projected onto sphere.
    std::vector<glm::vec3> dual(nFaces);
    for (int i = 0; i < nFaces; ++i) {
        const auto& f = faces[i];
        dual[i] = glm::normalize(verts[f[0]] + verts[f[1]] + verts[f[2]]);
    }

    // For each icosphere edge, connect the two adjacent face centroids —
    // that is one edge of the dual (hex/pentagon) wireframe.
    std::unordered_map<uint64_t, std::pair<int,int>> edgeMap;
    edgeMap.reserve((size_t)nFaces * 2);
    for (int fi = 0; fi < nFaces; ++fi) {
        const auto& f = faces[fi];
        for (int e = 0; e < 3; ++e) {
            int a = f[e], b = f[(e + 1) % 3];
            if (a > b) std::swap(a, b);
            const uint64_t key = ((uint64_t)a << 32) | (uint64_t)b;
            auto [it, inserted] = edgeMap.emplace(key, std::make_pair(fi, -1));
            if (!inserted) it->second.second = fi;
        }
    }

    const glm::vec3 lineColor(0.92f, 0.88f, 0.72f);
    out.clear();
    out.reserve(edgeMap.size() * 2);
    for (const auto& [key, pair] : edgeMap) {
        if (pair.second >= 0) {
            out.push_back({dual[pair.first],  lineColor});
            out.push_back({dual[pair.second], lineColor});
        }
    }
}

void generateNormalLines(int subdivisions, float length,
                         std::vector<HexVertex>& out) {
    IcoVerts verts = buildBaseVerts();
    IcoFaces faces = buildBaseFaces();
    for (int i = 0; i < subdivisions; ++i)
        subdivideOnce(verts, faces);

    const int nFaces = (int)faces.size();
    const glm::vec3 normalColor(0.2f, 0.9f, 0.9f);

    out.clear();
    out.reserve(nFaces * 2);
    for (int i = 0; i < nFaces; ++i) {
        const auto& f = faces[i];
        const glm::vec3 centre = glm::normalize(verts[f[0]] + verts[f[1]] + verts[f[2]]);
        out.push_back({centre,                    normalColor});
        out.push_back({centre * (1.0f + length),  normalColor});
    }
}

} // namespace HexGrid
