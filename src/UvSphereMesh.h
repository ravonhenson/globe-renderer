#pragma once

#include "Vertex.h"

#include <cstdint>
#include <vector>

// Builds a UV sphere: a latitude/longitude grid of vertices, two triangles
// per grid cell. The texture seam (texU wraps from 1 back to 0) and the
// poles (texU is otherwise undefined) are both handled the same way: the
// seam columns and each pole row are full rows/columns of distinct vertices
// that share a position but span texU = 0..1, so every triangle samples a
// contiguous strip of the texture. Triangles touching a pole have two
// corners that coincide in space (same position, different texU) and are
// simply degenerate/zero-area - they contribute nothing to the rendered
// image but keep the grid uniform.
void buildUvSphereMesh(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices);
