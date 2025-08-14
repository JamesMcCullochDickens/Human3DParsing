#version 330 core

out vec4 FragColor; // Output variable for fragment color

void main() {
    FragColor = vec4(abs(normalize(gl_FragCoord.xyz / gl_FragCoord.w)), 1.0);
}