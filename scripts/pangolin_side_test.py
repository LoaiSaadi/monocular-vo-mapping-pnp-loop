import os

# Windows: make sure the Pangolin/GL DLLs can be found
os.add_dll_directory(r"C:\tools\vcpkg\installed\x64-windows\bin")

# Import pangolin (works if you made the HW3 shim pangolin.py)
# Otherwise it falls back to pypangolin
try:
    import pangolin
except Exception:
    import pypangolin as pangolin

import OpenGL.GL as gl


def draw_wire_cube(size=1.0):
    """Draw a wireframe cube centered at origin."""
    s = size / 2.0
    v = [
        (-s, -s, -s), ( s, -s, -s), ( s,  s, -s), (-s,  s, -s),
        (-s, -s,  s), ( s, -s,  s), ( s,  s,  s), (-s,  s,  s),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    ]

    gl.glLineWidth(2.0)
    gl.glColor3f(0.9, 0.9, 0.9)
    gl.glBegin(gl.GL_LINES)
    for a, b in edges:
        gl.glVertex3f(*v[a])
        gl.glVertex3f(*v[b])
    gl.glEnd()


def draw_axes(length=1.5):
    """Draw XYZ axes."""
    gl.glLineWidth(3.0)
    gl.glBegin(gl.GL_LINES)

    # X (red)
    gl.glColor3f(1.0, 0.2, 0.2)
    gl.glVertex3f(0.0, 0.0, 0.0)
    gl.glVertex3f(length, 0.0, 0.0)

    # Y (green)
    gl.glColor3f(0.2, 1.0, 0.2)
    gl.glVertex3f(0.0, 0.0, 0.0)
    gl.glVertex3f(0.0, length, 0.0)

    # Z (blue)
    gl.glColor3f(0.2, 0.2, 1.0)
    gl.glVertex3f(0.0, 0.0, 0.0)
    gl.glVertex3f(0.0, 0.0, length)

    gl.glEnd()


def main():
    pangolin.CreateWindowAndBind("Pangolin Cube Viewer", 960, 720)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Camera / projection
    s_cam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(960, 720, 800, 800, 480, 360, 0.1, 1000),
        pangolin.ModelViewLookAt(3, 3, 3,   0, 0, 0,   0, 1, 0)
    )

    # Interactive 3D handler (mouse controls)
    handler = pangolin.Handler3D(s_cam)

    # Viewport
    d_cam = pangolin.CreateDisplay()
    d_cam.SetBounds(0.0, 1.0, 0.0, 1.0, -960 / 720)
    d_cam.SetHandler(handler)

    while not pangolin.ShouldQuit():
        gl.glClearColor(0.15, 0.15, 0.15, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        d_cam.Activate(s_cam)

        draw_axes(2.0)
        draw_wire_cube(1.5)

        pangolin.FinishFrame()


if __name__ == "__main__":
    main()