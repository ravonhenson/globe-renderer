#pragma once

#include "Vertex.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

// Dynamic, camera-distance-driven triangular quadtree mesh for the globe.
//
// Each of the 20 base icosahedron faces is the root of a quadtree; nodes
// near the camera recursively split into 4 children (the same
// edge-midpoint "triforce" split used by the static icosphere subdivision)
// while distant nodes stay coarse. Each leaf node is tessellated into a
// fixed-resolution grid, with the same per-triangle texU seam-unwrap and
// pole-averaging used by the static icosphere so the Natural Earth texture
// still maps cleanly at the antimeridian and poles, plus a skirt around its
// border (vertices pulled slightly toward the sphere center) to hide cracks
// where neighboring leaves are at different LOD levels.
namespace QuadtreeMesh {

// Tessellation resolution of each leaf patch: an NxN barycentric grid,
// giving (N+1)(N+2)/2 interior sample points and N^2 triangles.
constexpr uint32_t kGridSize = 8;

// Maximum quadtree depth (4^kMaxDepth finer than a root face at the deepest
// leaves). Each leaf is itself tessellated into a kGridSize x kGridSize
// grid, so the *effective* resolution at the deepest leaves is
// kMaxDepth + log4(kGridSize^2) levels finer than a root face.
constexpr uint32_t kMaxDepth = 6;

// A node splits into 4 children when the camera is closer than its edge
// length times this factor (in unit-sphere units). Larger values push LOD
// transitions farther from the camera (more detail, more triangles).
constexpr float kLodDistanceFactor = 10.0f;

// Skirt vertices are pulled toward the sphere center by this fraction of
// the radius, forming a small wall around each leaf to hide LOD cracks.
constexpr float kSkirtDepth = 0.01f;

// Safety cap on the total number of leaves a single selection can produce;
// also used by GlobeApp to size the dynamic vertex/index buffers. With
// kMaxDepth/kLodDistanceFactor above, the camera distance range from
// kMaxCameraDistance down to kMinCameraDistance produces at most roughly
// 5000 leaves (at kMinCameraDistance), so this leaves comfortable headroom.
constexpr size_t kMaxLeaves = 8192;

// selectRecursive's kMaxLeaves check happens before recursing, so once the
// cap is hit a single call can still push up to kMaxDepth + 1 extra leaves
// (one per level of an in-progress recursion) before every subsequent call
// observes leaves.size() >= kMaxLeaves. Buffers are sized for this many
// leaves beyond kMaxLeaves so an overshoot can never exceed capacity.
constexpr size_t kLeafOvershootMargin = kMaxDepth + 1;

constexpr size_t kGridPointCount = (kGridSize + 1) * (kGridSize + 2) / 2;
constexpr size_t kBoundaryPointCount = 3 * kGridSize;
constexpr size_t kInteriorTriangleCount = kGridSize * kGridSize;
constexpr size_t kSkirtTriangleCount = 2 * kBoundaryPointCount;

// Generous per-leaf upper bounds (interior grid + skirt top/bottom rims +
// margin for seam/pole texU duplicates) used to size GlobeApp's buffers.
constexpr size_t kMaxVerticesPerLeaf = kGridPointCount + 2 * kBoundaryPointCount + 16;
constexpr size_t kMaxIndicesPerLeaf = 3 * (kInteriorTriangleCount + kSkirtTriangleCount);

constexpr size_t kMaxVertices = (kMaxLeaves + kLeafOvershootMargin) * kMaxVerticesPerLeaf;
constexpr size_t kMaxIndices = (kMaxLeaves + kLeafOvershootMargin) * kMaxIndicesPerLeaf;

// A quadtree leaf: the unit-sphere positions of its 3 corners and its depth
// in the quadtree (0 = a base icosahedron face).
struct Patch {
    glm::vec3 a;
    glm::vec3 b;
    glm::vec3 c;
    uint32_t depth;

    bool operator==(const Patch& other) const {
        return a == other.a && b == other.b && c == other.c && depth == other.depth;
    }
};

// Walks the 20 base icosahedron faces, recursively splitting nodes that are
// closer to `cameraObjectPos` (in the mesh's own object space) than their
// edge length times kLodDistanceFactor, down to kMaxDepth. Returns the
// resulting leaf nodes.
std::vector<Patch> selectLeafPatches(const glm::vec3& cameraObjectPos);

// Tessellates and concatenates the given leaves into a single vertex/index
// buffer, ready for upload.
void generateMesh(const std::vector<Patch>& leaves, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices);

} // namespace QuadtreeMesh
