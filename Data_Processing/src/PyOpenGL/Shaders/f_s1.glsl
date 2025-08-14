#version 330 core

in vec3 fragment_color;
out vec4 color;

void main() {
    color = vec4(fragment_color, 1.0);
}