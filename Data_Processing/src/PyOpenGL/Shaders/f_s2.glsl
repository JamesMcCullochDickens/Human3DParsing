#version 330 core

in vec2 fragment_uv;
out vec4 color;

uniform sampler2D texture_sampler;

void main() {
    vec4 sampledColor = texture(texture_sampler, fragment_uv);
    color = vec4(sampledColor.rgb, sampledColor.a);
}