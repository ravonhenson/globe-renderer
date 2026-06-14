#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(binding = 2) uniform sampler2D heightSampler;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec2 fragTexCoord;

// Real elevation is tiny relative to Earth's radius (Everest is ~0.14%), so
// it is exaggerated here to be visible at globe scale.
const float kEarthRadiusMeters = 6371000.0;
const float kHeightExaggeration = 20.0;

void main() {
    float heightMeters = max(texture(heightSampler, inTexCoord).r, 0.0);
    float displacement = heightMeters / kEarthRadiusMeters * kHeightExaggeration;
    vec3 displacedPosition = inPosition + inNormal * displacement;

    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(displacedPosition, 1.0);
    fragNormal = mat3(ubo.model) * inNormal;
    fragTexCoord = inTexCoord;
}
