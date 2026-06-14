#pragma once

#include "Vertex.h"

#include <cstdint>
#include <vector>

// Builds an icosphere: a regular icosahedron, recursively subdivided and
// re-projected onto the unit sphere, giving a near-uniform triangle/vertex
// distribution with no pole pinching. Per-vertex texture coordinates are
// derived from each vertex's longitude/latitude using the same convention
// as the UV sphere it replaces, so the Natural Earth equirectangular
// texture maps onto it identically.
//
// The icosahedron is oriented so that exactly one vertex sits at each pole
// (x == z == 0), where longitude - and therefore texU - is undefined.
// Triangles that touch a pole, or that straddle the texU = 0/1 antimeridian
// seam, have their affected corners duplicated (same position, different
// texCoord) so every triangle samples a contiguous, non-degenerate patch of
// the texture - mirroring how the UV sphere handles its pole rows and seam
// columns.
void buildIcosphereMesh(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices);
