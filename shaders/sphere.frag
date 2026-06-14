#version 450

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 normal = normalize(fragNormal);
    vec3 lightDir = normalize(vec3(-0.4, 0.6, 0.8));

    float diffuse = max(dot(normal, lightDir), 0.0);
    float ambient = 0.25;
    float lighting = ambient + diffuse * (1.0 - ambient);

    vec3 color = texture(texSampler, fragTexCoord).rgb;
    outColor = vec4(color * lighting, 1.0);
}
