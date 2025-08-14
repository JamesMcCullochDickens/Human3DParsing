from OpenGL.GL import *
from OpenGL.GLUT import glutInit, glutCreateWindow, glutInitDisplayMode, GLUT_DOUBLE, GLUT_RGB
import glfw

# Initialize an OpenGL context using GLFW
def init_opengl_context():
    if not glfw.init():
        raise Exception("GLFW initialization failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    # Create a hidden window to establish an OpenGL context
    window = glfw.create_window(1, 1, "OpenGL Info", None, None)
    glfw.make_context_current(window)
    return window

# Print OpenGL and GLSL details
def print_opengl_details():
    print("=== OpenGL and GLSL Details ===")
    print("OpenGL Version:", glGetString(GL_VERSION).decode())
    print("GLSL Version:", glGetString(GL_SHADING_LANGUAGE_VERSION).decode())
    print("OpenGL Vendor:", glGetString(GL_VENDOR).decode())
    print("OpenGL Renderer:", glGetString(GL_RENDERER).decode())

    # Get major and minor version numbers
    major = glGetIntegerv(GL_MAJOR_VERSION)
    minor = glGetIntegerv(GL_MINOR_VERSION)
    print(f"OpenGL Version (Major.Minor): {major}.{minor}")

    # Extensions supported by OpenGL
    num_extensions = glGetIntegerv(GL_NUM_EXTENSIONS)
    extensions = [glGetStringi(GL_EXTENSIONS, i).decode() for i in range(num_extensions)]
    print("Supported Extensions:", extensions)

    # Context flags and profile mask
    context_flags = glGetIntegerv(GL_CONTEXT_FLAGS)
    profile_mask = glGetIntegerv(GL_CONTEXT_PROFILE_MASK)
    print("Context Flags:", context_flags)
    print("Profile Mask:", profile_mask)

    # Limits and capabilities (based on OpenGL core profile)
    max_vertex_attribs = glGetIntegerv(GL_MAX_VERTEX_ATTRIBS)
    max_uniform_blocks = glGetIntegerv(GL_MAX_UNIFORM_BUFFER_BINDINGS)
    max_texture_units = glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS)
    max_combined_texture_units = glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS)
    max_texture_size = glGetIntegerv(GL_MAX_TEXTURE_SIZE)
    max_3d_texture_size = glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE)
    max_elements_vertices = glGetIntegerv(GL_MAX_ELEMENTS_VERTICES)
    max_elements_indices = glGetIntegerv(GL_MAX_ELEMENTS_INDICES)

    print("\n=== OpenGL Limits and Capabilities ===")
    print("Max Vertex Attributes:", max_vertex_attribs)
    print("Max Uniform Buffer Bindings:", max_uniform_blocks)
    print("Max Texture Units (Fragment Shader):", max_texture_units)
    print("Max Combined Texture Units:", max_combined_texture_units)
    print("Max Texture Size:", max_texture_size)
    print("Max 3D Texture Size:", max_3d_texture_size)
    print("Max Elements Vertices:", max_elements_vertices)
    print("Max Elements Indices:", max_elements_indices)

    # Framebuffer parameters
    max_draw_buffers = glGetIntegerv(GL_MAX_DRAW_BUFFERS)
    max_color_attachments = glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS)
    print("Max Draw Buffers:", max_draw_buffers)
    print("Max Color Attachments:", max_color_attachments)

    # Close the window and terminate GLFW
    glfw.destroy_window(window)
    glfw.terminate()

# Main function
if __name__ == "__main__":
    window = init_opengl_context()
    print_opengl_details()
