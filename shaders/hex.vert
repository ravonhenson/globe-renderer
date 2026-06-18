#version 450

layout(push_constant) uniform PushConstants {
    mat4 mvp;
} pc;

layout(set = 0, binding = 2) uniform sampler2D heightSampler;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

const float PI = 3.14159265358979323846;
const float kEarthRadiusMeters = 6371000.0;
const float kHeightExaggeration = 20.0;
// Small constant lift to keep hex lines above the displaced terrain surface
// and avoid z-fighting. Same unit-sphere scale as the displacement.
const float kSurfaceBias = 0.002;

void main() {
    // Match the UV convention in QuadtreeMesh::computeUV:
    //   texU = 1 - atan2(z, x) / (2*PI), wrapped to [0, 1)
    //   texV = acos(y) / PI
    float texV = acos(clamp(inPosition.y, -1.0, 1.0)) / PI;
    float texU = mod(1.0 - atan(inPosition.z, inPosition.x) / (2.0 * PI), 1.0);

    float heightMeters = max(texture(heightSampler, vec2(texU, texV)).r, 0.0);
    float displacement = heightMeters / kEarthRadiusMeters * kHeightExaggeration;

    // inPosition is already on the unit sphere so it is its own normal.
    vec3 displaced = inPosition * (1.0 + displacement + kSurfaceBias);

    gl_Position = pc.mvp * vec4(displaced, 1.0);
    fragColor = inColor;
}
