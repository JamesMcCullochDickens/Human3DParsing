import sys
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import glfw


# Vertex Shader code
vertex_shader_code = """
#version 330
layout(location = 0) in vec2 position;

void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

# Fragment Shader code
fragment_shader_code = """
#version 330
out vec4 fragColor;

void main()
{
    fragColor = vec4(1.0, 0.0, 0.0, 1.0); // Red color
}
"""

def create_shader_program():
    vertex_shader = compileShader(vertex_shader_code, GL_VERTEX_SHADER)
    fragment_shader = compileShader(fragment_shader_code, GL_FRAGMENT_SHADER)
    shader_program = compileProgram(vertex_shader, fragment_shader)
    return shader_program

def main():
    # Initialize GLFW
    if not glfw.init():
        return

    # Create a window
    window = glfw.create_window(800, 600, "OpenGL Lines", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)

    # Define vertices for the lines
    vertices = np.array([
        -0.5,  0.5,  # Top left
         0.5,  0.5,  # Top right
         0.5, -0.5,  # Bottom right
        -0.5, -0.5,  # Bottom left
        -0.5,  0.5,  # Close the top left to top right line
         0.5, -0.5,  # Close the bottom left to bottom right line
    ], dtype='float32')

    # Create a Vertex Buffer Object (VBO) and Vertex Array Object (VAO)
    VBO = glGenBuffers(1)
    VAO = glGenVertexArrays(1)

    glBindVertexArray(VAO)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # Define the layout of the vertex data
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    # Compile shaders
    shader_program = create_shader_program()

    # Set the line width
    glLineWidth(10.0)

    # Main loop
    while not glfw.window_should_close(window):
        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT)

        # Use the shader program
        glUseProgram(shader_program)

        # Bind the VAO and draw the lines
        glBindVertexArray(VAO)
        glDrawArrays(GL_LINES, 0, len(vertices) // 2)  # Draw lines
        glBindVertexArray(0)

        # Swap buffers and poll for events
        glfw.swap_buffers(window)
        glfw.poll_events()

    # Clean up
    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glDeleteProgram(shader_program)

    glfw.terminate()

if __name__ == "__main__":
    main()