from typing import Optional
import numpy as np


def _win_add_vcpkg_dlls():
    import sys
    import os
    if sys.platform.startswith("win"):
        dll_dir = r"C:\tools\vcpkg\installed\x64-windows\bin"
        try:
            os.add_dll_directory(dll_dir)
        except (FileNotFoundError, OSError):
            pass


class PangolinViewer:
    def __init__(self, w: int = 1024, h: int = 768, title: str = "Trajectory + 3D Map (Pangolin)"):
        self.ok = False
        self.should_quit = False

        try:
            _win_add_vcpkg_dlls()
            import pangolin
            import OpenGL.GL as gl

            self.pangolin = pangolin
            self.gl = gl

            pangolin.CreateWindowAndBind(title, w, h)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glClearColor(0.10, 0.10, 0.12, 1.0)

            self.s_cam = pangolin.OpenGlRenderState(
                pangolin.ProjectionMatrix(w, h, 520, 520, w / 2.0, h / 2.0, 0.05, 5000),
                pangolin.ModelViewLookAt(0, -8, -8, 0, 0, 0, 0, -1, 0)
            )
            self.handler = pangolin.Handler3D(self.s_cam)

            self.d_cam = pangolin.CreateDisplay()
            aspect = -w / float(h)
            try:
                self.d_cam.SetBounds(0.0, 1.0, 0.0, 1.0, aspect)
            except TypeError:
                self.d_cam.SetBounds(
                    pangolin.Attach(0.0), pangolin.Attach(1.0),
                    pangolin.Attach(0.0), pangolin.Attach(1.0),
                    aspect
                )
            self.d_cam.SetHandler(self.handler)

            self.ok = True
            print("[INFO] Pangolin viewer enabled.")

        except Exception as e:
            print(f"[ERROR] Pangolin init failed: {e!r}")
            self.ok = False

    def _draw_camera_frustum(self, t_world_cam: np.ndarray, scale: float = 0.6) -> None:
        gl = self.gl

        if t_world_cam is None:
            return

        m = np.asarray(t_world_cam, dtype=np.float64).T.copy()

        gl.glPushMatrix()
        gl.glMultMatrixd(m)

        axis_len = 0.35 * scale

        gl.glLineWidth(2.0)
        gl.glBegin(gl.GL_LINES)

        gl.glColor3f(1.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(axis_len, 0.0, 0.0)

        gl.glColor3f(0.0, 1.0, 0.0)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, axis_len, 0.0)

        gl.glColor3f(0.0, 0.4, 1.0)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 0.0, axis_len)

        gl.glEnd()

        z = 0.8 * scale
        w = 0.5 * scale
        h = 0.35 * scale

        p0 = (0.0, 0.0, 0.0)
        p1 = (w, h, z)
        p2 = (w, -h, z)
        p3 = (-w, -h, z)
        p4 = (-w, h, z)

        gl.glColor3f(1.0, 1.0, 0.0)
        gl.glLineWidth(2.0)

        gl.glBegin(gl.GL_LINES)
        for p in (p1, p2, p3, p4):
            gl.glVertex3f(*p0)
            gl.glVertex3f(*p)
        gl.glEnd()

        gl.glBegin(gl.GL_LINE_LOOP)
        gl.glVertex3f(*p1)
        gl.glVertex3f(*p2)
        gl.glVertex3f(*p3)
        gl.glVertex3f(*p4)
        gl.glEnd()

        gl.glPopMatrix()

    def update(self,
               points_xyz: np.ndarray,
               traj_raw: np.ndarray,
               traj_corr: np.ndarray,
               current_pose_corr: Optional[np.ndarray] = None,
               flip_y_for_display: bool = True,
               max_points_draw: int = 20000) -> None:
        if not self.ok:
            return

        pangolin = self.pangolin
        gl = self.gl

        if pangolin.ShouldQuit():
            self.should_quit = True
            return

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self.d_cam.Activate(self.s_cam)

        if hasattr(pangolin, "glDrawAxis"):
            pangolin.glDrawAxis(1.0)

        pts = np.asarray(points_xyz, dtype=np.float64) if points_xyz is not None else np.zeros((0, 3), np.float64)
        raw = np.asarray(traj_raw, dtype=np.float64) if traj_raw is not None else np.zeros((0, 3), np.float64)
        cor = np.asarray(traj_corr, dtype=np.float64) if traj_corr is not None else np.zeros((0, 3), np.float64)

        if flip_y_for_display:
            if len(pts):
                pts = pts.copy()
                pts[:, 1] *= -1.0
            if len(raw):
                raw = raw.copy()
                raw[:, 1] *= -1.0
            if len(cor):
                cor = cor.copy()
                cor[:, 1] *= -1.0

        if len(pts) > max_points_draw:
            idx = np.random.choice(len(pts), max_points_draw, replace=False)
            pts = pts[idx]

        if len(pts) > 0:
            gl.glPointSize(2.0)
            gl.glColor3f(0.70, 0.70, 0.70)
            gl.glBegin(gl.GL_POINTS)
            for x, y, z in pts:
                gl.glVertex3f(float(x), float(y), float(z))
            gl.glEnd()

        if len(cor) >= 2:
            gl.glLineWidth(3.0)
            gl.glColor3f(0.0, 1.0, 0.0)
            gl.glBegin(gl.GL_LINE_STRIP)
            for x, y, z in cor:
                gl.glVertex3f(float(x), float(y), float(z))
            gl.glEnd()

        if current_pose_corr is not None:
            t_draw = np.asarray(current_pose_corr, dtype=np.float64).copy()
            if flip_y_for_display:
                t_draw[1, 3] *= -1.0
            self._draw_camera_frustum(t_draw, scale=0.8)

        pangolin.FinishFrame()