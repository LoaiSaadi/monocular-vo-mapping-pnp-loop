import os
import sys
import csv
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ============================================================
# HW3 - Monocular VO + Epipolar error + PointCloud + PnP + Loop
# ============================================================
# Updates in this version (to match HW PDF more strictly):
# - Epipolar error is computed in TWO explicit stages:
#   (A) "Descriptor stage" BEFORE geometry: Fundamental matrix via 8-point (no RANSAC)
#   (B) AFTER geometry: Fundamental matrix via RANSAC + inlier mask
# - FRAME now has "Pose" field (camera->world) per HW wording.
# - PnP refinement optimizer selectable: LM (OpenCV) or GN (numeric) or none
# - CSV metrics include epi_med_desc / epi_med_geom_in + means
#
# Notes:
# - Intrinsics K are estimated from image size (fx = 0.9*w, fy=fx, cx=w/2, cy=h/2)
# - Monocular scale is unknown: translation direction is normalized for VO integration
# - Loop-closure is a lightweight "snap" correction (pose correction only; no full pose-graph/BA)

# ----------------------------
# Windows: ensure Pangolin DLLs are reachable (vcpkg)
# ----------------------------
def _win_add_vcpkg_dlls():
    if sys.platform.startswith("win"):
        dll_dir = r"C:\tools\vcpkg\installed\x64-windows\bin"
        try:
            os.add_dll_directory(dll_dir)
        except (FileNotFoundError, OSError):
            # Pangolin may still work if PATH already includes it
            pass

# Ensure DLL dir is registered as early as possible
_win_add_vcpkg_dlls()


# ----------------------------
# Dataset helpers (TUM format)
# ----------------------------
def load_tum_list(txt_path: str) -> List[Tuple[float, str]]:
    items = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ts, rel = line.split()[:2]
            items.append((float(ts), rel))
    return items


def get_image_path(dataset_dir: str, rel: str) -> str:
    c1 = os.path.join(dataset_dir, rel)
    c2 = os.path.join(dataset_dir, "rgb", os.path.basename(rel))
    return c1 if os.path.exists(c1) else c2


def build_k_from_image_size(w: int, h: int) -> np.ndarray:
    fx = 0.9 * w
    fy = fx
    cx = w / 2.0
    cy = h / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)


# ----------------------------
# Data structures (HW-style)
# ----------------------------
@dataclass
class FRAME:
    Id: int
    Timestamp: float
    Pose: np.ndarray                  # 4x4 camera->world (raw VO chain)
    KeyPoints: list
    Descriptors: np.ndarray
    ImageGray: np.ndarray
    ImageBgr: np.ndarray
    Processed: bool = True


@dataclass
class Point:
    Id: int
    point: np.ndarray                  # (3,) in corrected/global world
    frames: list = field(default_factory=list)   # frame ids that observe it
    des: np.ndarray = None             # ORB descriptor (32,)
    color_bgr: tuple = (255, 255, 255)


@dataclass
class Map:
    points: list = field(default_factory=list)   # list[Point]
    frames: list = field(default_factory=list)   # list[FRAME]


# ----------------------------
# SE(3) utilities
# ----------------------------
def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.astype(np.float64)
    T[:3, 3] = t.reshape(3).astype(np.float64)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -(R.T @ t)
    return Ti


def rel_T_c2_c1(T_w_c1: np.ndarray, T_w_c2: np.ndarray) -> np.ndarray:
    # T_c2_c1 = inv(T_w_c2) @ T_w_c1
    return invert_T(T_w_c2) @ T_w_c1


def apply_T_to_points(T: np.ndarray, X: np.ndarray) -> np.ndarray:
    # X: Nx3
    Xh = np.hstack([X, np.ones((len(X), 1), dtype=np.float64)])
    Yh = (T @ Xh.T).T
    return Yh[:, :3]


# ----------------------------
# Epipolar Sampson error
# ----------------------------
def sampson_errors(F: Optional[np.ndarray], pts1: np.ndarray, pts2: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    pts1, pts2: Nx2 in pixels (float64)
    Returns: N vector of Sampson errors.
    """
    if F is None or len(pts1) == 0:
        return np.array([], dtype=np.float64)

    x1 = np.hstack([pts1, np.ones((len(pts1), 1), dtype=np.float64)])  # Nx3
    x2 = np.hstack([pts2, np.ones((len(pts2), 1), dtype=np.float64)])  # Nx3

    Fx1 = (F @ x1.T).T
    Ftx2 = (F.T @ x2.T).T
    x2tFx1 = np.sum(x2 * Fx1, axis=1)

    denom = Fx1[:, 0]**2 + Fx1[:, 1]**2 + Ftx2[:, 0]**2 + Ftx2[:, 1]**2
    return (x2tFx1**2) / (denom + eps)


def fundamental_8point(pts1: np.ndarray, pts2: np.ndarray) -> Optional[np.ndarray]:
    """
    Fundamental matrix BEFORE geometry stage:
    Use 8-point (no RANSAC), so epipolar error represents "descriptor matches stage".
    """
    if pts1 is None or pts2 is None or len(pts1) < 8:
        return None
    F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    if F is None:
        return None
    if F.shape != (3, 3):
        try:
            F = F.reshape(3, 3)
        except Exception:
            return None
    return F


# ----------------------------
# Projection utilities + RMSE
# ----------------------------
def project_points(K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, Xw: np.ndarray) -> np.ndarray:
    uv, _ = cv2.projectPoints(Xw.astype(np.float64), rvec, tvec, K, None)
    return uv.reshape(-1, 2)


def reproj_rmse(K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, Xw: np.ndarray, uv: np.ndarray) -> float:
    pred = project_points(K, rvec, tvec, Xw)
    err = np.linalg.norm(pred - uv, axis=1)
    return float(np.sqrt(np.mean(err**2))) if len(err) else float("nan")


# ----------------------------
# Pose refinement (GN numeric)
# ----------------------------
def refine_pose_gn_numeric(K: np.ndarray,
                           rvec: np.ndarray,
                           tvec: np.ndarray,
                           Xw: np.ndarray,
                           uv: np.ndarray,
                           iters: int = 7,
                           eps: float = 1e-6,
                           damping: float = 1e-3):
    """
    Simple Gauss-Newton refinement of a PnP pose (numeric Jacobian).
    """
    rvec = rvec.reshape(3, 1).astype(np.float64).copy()
    tvec = tvec.reshape(3, 1).astype(np.float64).copy()

    def residuals(params: np.ndarray) -> np.ndarray:
        rv = params[:3].reshape(3, 1)
        tv = params[3:].reshape(3, 1)
        pred = project_points(K, rv, tv, Xw)  # Nx2
        return (uv - pred).reshape(-1)        # 2N

    params = np.vstack([rvec, tvec]).reshape(-1, 1)  # 6x1

    for _ in range(iters):
        r0 = residuals(params)
        if len(r0) < 12:
            break

        J = np.zeros((len(r0), 6), dtype=np.float64)
        for j in range(6):
            p2 = params.copy()
            p2[j, 0] += eps
            r2 = residuals(p2)
            J[:, j] = (r2 - r0) / eps

        A = J.T @ J + damping * np.eye(6)
        b = -J.T @ r0.reshape(-1, 1)
        try:
            dx = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break

        params = params + dx

    rvec_ref = params[:3].reshape(3, 1)
    tvec_ref = params[3:].reshape(3, 1)
    return rvec_ref, tvec_ref


# ----------------------------
# Map I/O
# ----------------------------
def write_ply(path: str, points: List[Point]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar blue\nproperty uchar green\nproperty uchar red\n")
        f.write("end_header\n")
        for p in points:
            x, y, z = p.point.tolist()
            b, g, r = p.color_bgr
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(b)} {int(g)} {int(r)}\n")


# ----------------------------
# Triangulation (keyframe -> current)
# ----------------------------
def add_points_from_keyframe(world_map: Map,
                             keyframe: FRAME,
                             cur_frame: FRAME,
                             T_w_c_kf: np.ndarray,
                             T_w_c_cur: np.ndarray,
                             K: np.ndarray,
                             max_abs_coord: float = 500.0,
                             max_depth: float = 400.0,
                             reproj_thresh_px: float = 10.0,
                             min_depth: float = 0.05,
                             max_new: int = 800,
                             pid_start: int = 0) -> int:
    """
    Adds new 3D points triangulated between keyframe and current frame.
    - Descriptors are copied from the keyframe feature.
    - Points are stored in corrected/global world coordinates.
    Returns updated pid_start.
    """
    if keyframe.Descriptors is None or cur_frame.Descriptors is None:
        return pid_start

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(keyframe.Descriptors, cur_frame.Descriptors, k=2)

    good = []
    for m, n in knn:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 30:
        return pid_start

    pts1_all = np.float32([keyframe.KeyPoints[m.queryIdx].pt for m in good])
    pts2_all = np.float32([cur_frame.KeyPoints[m.trainIdx].pt for m in good])

    F, fmask = cv2.findFundamentalMat(pts1_all, pts2_all, cv2.FM_RANSAC, 1.5, 0.999)
    if F is None or fmask is None:
        return pid_start
    fmask = fmask.ravel().astype(bool)

    good = [m for m, keep in zip(good, fmask) if keep]
    if len(good) < 20:
        return pid_start

    if len(good) > max_new:
        idx = np.random.choice(len(good), max_new, replace=False)
        good = [good[k] for k in idx]

    pts1 = np.float32([keyframe.KeyPoints[m.queryIdx].pt for m in good])
    pts2 = np.float32([cur_frame.KeyPoints[m.trainIdx].pt for m in good])

    # Relative pose (keyframe -> current) in camera coords
    T_c_cur_c_kf = rel_T_c2_c1(T_w_c_kf, T_w_c_cur)
    R = T_c_cur_c_kf[:3, :3]
    t = T_c_cur_c_kf[:3, 3:4]

    # Normalize points to camera coordinates
    pts1n = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    pts2n = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None).reshape(-1, 2)

    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, t])

    X_h = cv2.triangulatePoints(P1, P2, pts1n.T, pts2n.T).T
    X_kf = X_h[:, :3] / np.maximum(X_h[:, 3:4], 1e-12)

    # Cheirality: depth positive in both cameras
    z1 = X_kf[:, 2]
    X_cur = (R @ X_kf.T + t).T
    z2 = X_cur[:, 2]
    keep = (z1 > min_depth) & (z2 > min_depth) & (z1 < max_depth) & (z2 < max_depth)
    X_kf = X_kf[keep]
    pts1 = pts1[keep]
    pts2 = pts2[keep]
    good = [m for m, k in zip(good, keep) if k]

    if len(X_kf) < 10:
        return pid_start

    # Reprojection filter in pixel space
    P1_pix = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2_pix = K @ np.hstack([R, t])

    def reproj(Ppix: np.ndarray, Xcam: np.ndarray, uv: np.ndarray) -> np.ndarray:
        X_h2 = np.hstack([Xcam, np.ones((len(Xcam), 1))])
        x = (Ppix @ X_h2.T).T
        x = x[:, :2] / np.maximum(x[:, 2:3], 1e-12)
        return np.linalg.norm(x - uv, axis=1)

    e1 = reproj(P1_pix, X_kf, pts1)
    e2 = reproj(P2_pix, X_kf, pts2)
    keep2 = (e1 < reproj_thresh_px) & (e2 < reproj_thresh_px)
    X_kf = X_kf[keep2]
    pts1 = pts1[keep2]
    good = [m for m, k in zip(good, keep2) if k]

    if len(X_kf) < 10:
        return pid_start

    # Transform to corrected world
    X_w = (T_w_c_kf @ np.hstack([X_kf, np.ones((len(X_kf), 1))]).T).T[:, :3]

    # Add points with descriptor + color from keyframe
    for j in range(len(X_w)):
        x, y, z = X_w[j]
        if not np.isfinite([x, y, z]).all():
            continue
        if max(abs(x), abs(y), abs(z)) > max_abs_coord:
            continue

        u, v = pts1[j]
        ui, vi = int(round(u)), int(round(v))
        if 0 <= vi < keyframe.ImageBgr.shape[0] and 0 <= ui < keyframe.ImageBgr.shape[1]:
            color = tuple(int(c) for c in keyframe.ImageBgr[vi, ui])
        else:
            color = (255, 255, 255)

        desc = keyframe.Descriptors[good[j].queryIdx].copy()
        world_map.points.append(Point(
            Id=pid_start,
            point=X_w[j].astype(np.float64),
            frames=[keyframe.Id, cur_frame.Id],
            des=desc,
            color_bgr=color
        ))
        pid_start += 1

    return pid_start


# ----------------------------
# PnP localization + optimization
# ----------------------------
def localize_pnp(world_map: Map,
                 frame: FRAME,
                 K: np.ndarray,
                 min_corr: int = 50,
                 refine: str = "lm"):
    """
    Returns:
      (T_w_c, matches, inliers, rmse_before, rmse_after) or None
    refine: "lm" | "gn" | "none"
    """
    if len(world_map.points) < min_corr:
        return None
    if frame.Descriptors is None:
        return None

    map_desc = np.vstack([p.des for p in world_map.points])  # Nx32
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(map_desc, frame.Descriptors, k=2)

    good = []
    for m, n in knn:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < min_corr:
        return None

    Xw = np.float32([world_map.points[m.queryIdx].point for m in good])
    uv = np.float32([frame.KeyPoints[m.trainIdx].pt for m in good])

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        Xw, uv, K, None,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=6.0,
        confidence=0.999,
        iterationsCount=200
    )
    if (not ok) or (inliers is None) or (len(inliers) < 20):
        return None

    inliers = inliers.reshape(-1)
    Xw_in = Xw[inliers].astype(np.float64)
    uv_in = uv[inliers].astype(np.float64)

    rmse_before = reproj_rmse(K, rvec, tvec, Xw_in, uv_in)

    rvec_ref, tvec_ref = rvec, tvec
    if refine == "lm":
        if hasattr(cv2, "solvePnPRefineLM"):
            try:
                rvec_ref, tvec_ref = cv2.solvePnPRefineLM(Xw_in, uv_in, K, None, rvec, tvec)
            except cv2.error:
                rvec_ref, tvec_ref = rvec, tvec
        else:
            rvec_ref, tvec_ref = refine_pose_gn_numeric(K, rvec, tvec, Xw_in, uv_in, iters=7, damping=1e-3)
    elif refine == "gn":
        rvec_ref, tvec_ref = refine_pose_gn_numeric(K, rvec, tvec, Xw_in, uv_in, iters=7, damping=1e-3)

    rmse_after = reproj_rmse(K, rvec_ref, tvec_ref, Xw_in, uv_in)

    R, _ = cv2.Rodrigues(rvec_ref)
    T_c_w = make_T(R, tvec_ref)  # world->camera (Xc = R*Xw + t)
    T_w_c = invert_T(T_c_w)      # camera->world

    return T_w_c, len(good), len(inliers), rmse_before, rmse_after


# ----------------------------
# Lightweight loop closure
# ----------------------------
def detect_loop_closure(keyframes: List[FRAME],
                        cur_frame: FRAME,
                        K: np.ndarray,
                        min_gap: int = 40,
                        min_inliers: int = 220):
    """
    Returns (kf_id, T_c_cur_c_kf, inliers_count) or None.
    """
    if cur_frame.Descriptors is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    best = None

    for kf in keyframes:
        if (cur_frame.Id - kf.Id) < min_gap:
            continue
        if kf.Descriptors is None:
            continue

        knn = bf.knnMatch(kf.Descriptors, cur_frame.Descriptors, k=2)
        good = []
        for m, n in knn:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) < (min_inliers // 2):
            continue

        pts1 = np.float32([kf.KeyPoints[m.queryIdx].pt for m in good])
        pts2 = np.float32([cur_frame.KeyPoints[m.trainIdx].pt for m in good])

        F, fmask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.5, 0.999)
        if F is None or fmask is None:
            continue
        fmask = fmask.ravel().astype(bool)
        inl = int(fmask.sum())
        if inl < min_inliers:
            continue

        pts1_in = pts1[fmask]
        pts2_in = pts2[fmask]

        E, _ = cv2.findEssentialMat(pts1_in, pts2_in, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            continue

        try:
            _, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, K)
        except cv2.error:
            continue

        T_c_cur_c_kf = make_T(R, t)

        if (best is None) or (inl > best[2]):
            best = (kf.Id, T_c_cur_c_kf, inl)

    return best


# ----------------------------
# 2D Trajectory visualization (raw vs corrected)
# ----------------------------
def draw_trajectory_xz(raw_positions, corr_positions, size=800, scale=20.0, center=None):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    if center is None:
        center = (size // 2, size // 2)

    def to_px(p):
        x, z = float(p[0]), float(p[2])
        px = int(x * scale + center[0])
        pz = int(z * scale + center[1])
        return px, pz

    for i in range(1, len(raw_positions)):
        a = to_px(raw_positions[i - 1])
        b = to_px(raw_positions[i])
        cv2.line(img, a, b, (0, 0, 255), 1)

    for i in range(1, len(corr_positions)):
        a = to_px(corr_positions[i - 1])
        b = to_px(corr_positions[i])
        cv2.line(img, a, b, (0, 255, 0), 2)

    return img


# ----------------------------
# Pangolin viewer (preferred by HW)
# ----------------------------
class PangolinViewer:
    def __init__(self, w=1024, h=768):
        self.ok = False
        self.should_quit = False
        self._w = w
        self._h = h

        try:
            _win_add_vcpkg_dlls()

            import pangolin
            import OpenGL.GL as gl

            self.pangolin = pangolin
            self.gl = gl

            pangolin.CreateWindowAndBind("trajectory_and_map_pangolin", w, h)
            gl.glEnable(gl.GL_DEPTH_TEST)

            # Not pure black -> easier to see if it's rendering
            gl.glClearColor(0.10, 0.10, 0.12, 1.0)

            self.s_cam = pangolin.OpenGlRenderState(
                pangolin.ProjectionMatrix(w, h, 520, 520, w / 2.0, h / 2.0, 0.05, 2000),
                pangolin.ModelViewLookAt(
                    0, -3, -3,   # camera position
                    0,  0,  0,   # look at
                    0, -1,  0    # up
                )
            )

            self.handler = pangolin.Handler3D(self.s_cam)
            # self.d_cam = pangolin.CreateDisplay()
            # self.d_cam.SetBounds(0.0, 1.0, 0.0, 1.0, -w / float(h))
            # self.d_cam.SetHandler(self.handler)
            self.d_cam = pangolin.CreateDisplay()
            aspect = -w / float(h)

            # pypangolin wants Attach objects (your case). Some bindings accept floats.
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

        except Exception as e:
            print(f"[WARN] Pangolin init failed: {e!r}")
            self.ok = False

    def update(self, points_xyz: np.ndarray, points_rgb: np.ndarray, traj_xyz: np.ndarray):
        if not self.ok:
            return

        pangolin = self.pangolin
        gl = self.gl

        if pangolin.ShouldQuit():
            self.should_quit = True
            return

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self.d_cam.Activate(self.s_cam)

        # Always draw an axis so you can confirm the window is alive
        if hasattr(pangolin, "glDrawAxis"):
            pangolin.glDrawAxis(1.0)

        pts = np.asarray(points_xyz, dtype=np.float64) if points_xyz is not None else np.zeros((0, 3), np.float64)
        rgb = np.asarray(points_rgb, dtype=np.float64) if points_rgb is not None else np.zeros((0, 3), np.float64)
        traj = np.asarray(traj_xyz, dtype=np.float64) if traj_xyz is not None else np.zeros((0, 3), np.float64)

        # Recenter for display ONLY (doesn't change your actual map)
        center = np.zeros(3, dtype=np.float64)
        if len(traj) > 0:
            center = traj[-1].copy()
        elif len(pts) > 0:
            center = np.mean(pts, axis=0)

        pts_disp = pts - center if len(pts) else pts
        traj_disp = traj - center if len(traj) else traj

        # Points
        if len(pts_disp) > 0:
            gl.glPointSize(2.0)
            gl.glBegin(gl.GL_POINTS)
            if len(rgb) == len(pts_disp):
                for (x, y, z), (b, g, r) in zip(pts_disp, rgb):
                    gl.glColor3f(float(r) / 255.0, float(g) / 255.0, float(b) / 255.0)
                    gl.glVertex3f(float(x), float(y), float(z))
            else:
                gl.glColor3f(1.0, 1.0, 1.0)
                for (x, y, z) in pts_disp:
                    gl.glVertex3f(float(x), float(y), float(z))
            gl.glEnd()

        # Trajectory
        if len(traj_disp) >= 2:
            gl.glLineWidth(2.0)
            gl.glColor3f(0.0, 1.0, 0.0)
            gl.glBegin(gl.GL_LINE_STRIP)
            for x, y, z in traj_disp:
                gl.glVertex3f(float(x), float(y), float(z))
            gl.glEnd()

        pangolin.FinishFrame()

    def close(self):
        pass


# ----------------------------
# Open3D viewer (fallback)
# ----------------------------
class Open3DViewer:
    def __init__(self):
        self.ok = False
        self.o3d = None
        try:
            import open3d as o3d
            self.o3d = o3d
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name="trajectory_and_map_open3d", width=960, height=720, visible=True)
            self.pcd = o3d.geometry.PointCloud()
            self.traj = o3d.geometry.LineSet()
            self.added = False
            self.ok = True
        except Exception:
            self.ok = False

    def update(self, points_xyz: np.ndarray, points_rgb: np.ndarray, traj_xyz: np.ndarray):
        if not self.ok:
            return

        o3d = self.o3d
        if len(points_xyz) > 0:
            self.pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
            self.pcd.colors = o3d.utility.Vector3dVector(np.clip(points_rgb.astype(np.float64) / 255.0, 0, 1))
        else:
            self.pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
            self.pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))

        if len(traj_xyz) >= 2:
            pts = traj_xyz.astype(np.float64)
            lines = np.array([[i, i + 1] for i in range(len(pts) - 1)], dtype=np.int32)
            self.traj.points = o3d.utility.Vector3dVector(pts)
            self.traj.lines = o3d.utility.Vector2iVector(lines)
        else:
            self.traj.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
            self.traj.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))

        if not self.added:
            self.vis.add_geometry(self.pcd)
            self.vis.add_geometry(self.traj)
            self.added = True
        else:
            self.vis.update_geometry(self.pcd)
            self.vis.update_geometry(self.traj)

        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        if self.ok:
            self.vis.destroy_window()


# ----------------------------
# Main
# ----------------------------
def main(dataset_dir: str,
         max_frames: int = 600,
         keyframe_interval: int = 20,
         pnp_every: int = 10,
         loop_every: int = 30,
         show_every: int = 5,
         viewer_mode: str = "auto",
         show_matches_window: bool = True,
         zero_origin_on_save: bool = True,
         pnp_refine: str = "lm"):

    rgb_list = load_tum_list(os.path.join(dataset_dir, "rgb.txt"))
    if len(rgb_list) < 2:
        raise RuntimeError("rgb.txt too short")

    project_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(project_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    out_ply = os.path.join(out_dir, "map_points.ply")
    out_traj_raw = os.path.join(out_dir, "trajectory_raw.txt")
    out_traj_corr = os.path.join(out_dir, "trajectory_corrected.txt")
    out_metrics = os.path.join(out_dir, "metrics.csv")

    metrics_f = open(out_metrics, "w", newline="", encoding="utf-8")
    fieldnames = [
        "frame",
        "matches_total",
        "inliers_ransac",
        "epi_med_desc",
        "epi_mean_desc",
        "epi_med_geom_in",
        "epi_mean_geom_in",
        "pnp_matches",
        "pnp_inliers",
        "pnp_rmse_before",
        "pnp_rmse_after",
        "loop_kf",
        "loop_inliers",
        "map_points"
    ]
    writer = csv.DictWriter(metrics_f, fieldnames=fieldnames)
    writer.writeheader()

    orb = cv2.ORB_create(nfeatures=8000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    world_map = Map(points=[], frames=[])
    pid = 0

    # Raw VO pose chain (camera->world)
    T_w_c_raw = np.eye(4, dtype=np.float64)

    # Correction transform applied on top of raw poses (for PnP + loop closure effect)
    T_world_corr = np.eye(4, dtype=np.float64)

    prev_frame: Optional[FRAME] = None
    keyframe: Optional[FRAME] = None
    keyframes_for_loop: List[FRAME] = []
    K: Optional[np.ndarray] = None

    raw_positions = []
    corr_positions = []
    raw_log = []
    corr_log = []

    viewer = None
    if viewer_mode in ("auto", "pangolin"):
        viewer = PangolinViewer()
        if viewer.ok:
            print("[INFO] Pangolin viewer enabled.")
        else:
            viewer = None
            if viewer_mode == "pangolin":
                print("[WARN] Pangolin requested but not available.")
            else:
                print("[INFO] Pangolin not available -> trying Open3D fallback viewer.")

    if viewer is None and viewer_mode in ("auto", "open3d"):
        viewer = Open3DViewer()
        if viewer.ok:
            print("[INFO] Open3D viewer enabled.")
        else:
            viewer = None
            if viewer_mode == "open3d":
                print("[WARN] Open3D requested but not available. Run: pip install open3d")
            else:
                print("[INFO] Open3D not available -> using only OpenCV windows (still saving PLY).")

    # Cached arrays for viewer (rebuild only every show_every; pangolin still updates every frame)
    viewer_xyz = np.zeros((0, 3), dtype=np.float64)
    viewer_rgb = np.zeros((0, 3), dtype=np.float64)
    viewer_traj = np.zeros((0, 3), dtype=np.float64)

    try:
        for i in range(min(max_frames, len(rgb_list))):
            ts, rel = rgb_list[i]
            img_path = get_image_path(dataset_dir, rel)

            img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            if K is None:
                h, w = img_gray.shape[:2]
                K = build_k_from_image_size(w, h)

            kp, des = orb.detectAndCompute(img_gray, None)
            if des is None or len(kp) < 120:
                continue

            frame = FRAME(
                Id=i,
                Timestamp=ts,
                Pose=T_w_c_raw.copy(),
                KeyPoints=kp,
                Descriptors=des,
                ImageGray=img_gray,
                ImageBgr=img_bgr
            )

            # Metrics init
            m_total = 0
            m_inl = 0
            epi_med_desc = float("nan")
            epi_mean_desc = float("nan")
            epi_med_geom_in = float("nan")
            epi_mean_geom_in = float("nan")

            pnp_m = 0
            pnp_inl = 0
            rmse_b = float("nan")
            rmse_a = float("nan")

            loop_kf = -1
            loop_inl = 0

            # Window 1: current frame with keypoints
            if (i % show_every) == 0:
                vis_kp = cv2.drawKeypoints(img_bgr, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.imshow("current_keypoints", vis_kp)

            matches_before_img = None
            matches_after_img = None

            # ----------------------------
            # VO step: match prev->cur, compute F/E, integrate pose
            # ----------------------------
            if prev_frame is not None:
                knn = matcher.knnMatch(prev_frame.Descriptors, frame.Descriptors, k=2)

                good = []
                for m, n in knn:
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

                if len(good) >= 20:
                    if show_matches_window and (i % show_every) == 0:
                        mshow = sorted(good, key=lambda x: x.distance)[:250]
                        matches_before_img = cv2.drawMatches(prev_frame.ImageBgr, prev_frame.KeyPoints,
                                                             frame.ImageBgr, frame.KeyPoints,
                                                             mshow, None,
                                                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

                    pts1_all = np.float32([prev_frame.KeyPoints[m.queryIdx].pt for m in good])
                    pts2_all = np.float32([frame.KeyPoints[m.trainIdx].pt for m in good])

                    # Epipolar error BEFORE geometry (8-point)
                    F_desc = fundamental_8point(pts1_all, pts2_all)
                    se_desc = sampson_errors(F_desc, pts1_all.astype(np.float64), pts2_all.astype(np.float64))
                    if len(se_desc):
                        epi_med_desc = float(np.median(se_desc))
                        epi_mean_desc = float(np.mean(se_desc))

                    # Geometry estimation (RANSAC) + inliers
                    F, fmask = cv2.findFundamentalMat(pts1_all, pts2_all, cv2.FM_RANSAC, 1.0, 0.999)
                    if F is not None and fmask is not None:
                        fmask = fmask.ravel().astype(bool)
                        inl = int(fmask.sum())

                        m_total = int(len(good))
                        m_inl = int(inl)

                        # Epipolar error AFTER geometry: only on inliers
                        pts1_in = pts1_all[fmask]
                        pts2_in = pts2_all[fmask]
                        se_geom_in = sampson_errors(F, pts1_in.astype(np.float64), pts2_in.astype(np.float64))
                        if len(se_geom_in):
                            epi_med_geom_in = float(np.median(se_geom_in))
                            epi_mean_geom_in = float(np.mean(se_geom_in))

                        # Show matches AFTER filtering
                        if show_matches_window and (i % show_every) == 0:
                            inlier_matches = [m for m, keep in zip(good, fmask) if keep]
                            inlier_matches = sorted(inlier_matches, key=lambda x: x.distance)[:250]
                            matches_after_img = cv2.drawMatches(prev_frame.ImageBgr, prev_frame.KeyPoints,
                                                                frame.ImageBgr, frame.KeyPoints,
                                                                inlier_matches, None,
                                                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

                        # Relative motion from epipolar geometry
                        if inl >= 12:
                            E, _ = cv2.findEssentialMat(pts1_in, pts2_in, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                            if E is not None:
                                try:
                                    _, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, K)
                                    tn = t / max(np.linalg.norm(t), 1e-12)  # direction only
                                    T_c2_c1 = make_T(R, tn)                 # c1 -> c2
                                    T_w_c_raw = prev_frame.Pose @ invert_T(T_c2_c1)
                                    frame.Pose = T_w_c_raw.copy()
                                except cv2.error:
                                    pass

            # Show before/after matches window + overlay epipolar error values
            if show_matches_window and (i % show_every) == 0 and (matches_before_img is not None or matches_after_img is not None):
                if matches_before_img is None:
                    matches_before_img = np.zeros_like(matches_after_img)
                if matches_after_img is None:
                    matches_after_img = np.zeros_like(matches_before_img)

                w = max(matches_before_img.shape[1], matches_after_img.shape[1])

                def pad_to(img, w_):
                    if img.shape[1] == w_:
                        return img
                    pad = np.zeros((img.shape[0], w_ - img.shape[1], 3), dtype=np.uint8)
                    return np.hstack([img, pad])

                top = pad_to(matches_before_img, w)
                bot = pad_to(matches_after_img, w)
                stack = np.vstack([top, bot])

                text1 = f"matches={m_total}  inliers={m_inl}"
                text2 = f"epi(desc) med={epi_med_desc:.3g} mean={epi_mean_desc:.3g} | epi(geom-in) med={epi_med_geom_in:.3g} mean={epi_mean_geom_in:.3g}"
                cv2.putText(stack, text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(stack, text2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.imshow("matches_before_after", stack)

            # Corrected pose (raw + correction)
            T_w_c_corr = T_world_corr @ frame.Pose

            # Initialize first keyframe
            if keyframe is None:
                keyframe = frame
                keyframes_for_loop.append(frame)

            # Triangulate map points between keyframe and current
            if frame.Id != keyframe.Id and len(world_map.points) < 8000:
                T_w_c_kf_corr = T_world_corr @ keyframe.Pose
                pid = add_points_from_keyframe(
                    world_map, keyframe, frame,
                    T_w_c_kf_corr, T_w_c_corr, K,
                    max_abs_coord=500.0,
                    max_depth=400.0,
                    reproj_thresh_px=10.0,
                    min_depth=0.05,
                    max_new=800,
                    pid_start=pid
                )

            # PnP relocalization every N frames
            if (frame.Id % pnp_every) == 0 and len(world_map.points) >= 80:
                res = localize_pnp(world_map, frame, K, min_corr=50, refine=pnp_refine)
                if res is not None:
                    T_w_c_pnp, n_matches, n_inliers, rb, ra = res
                    pnp_m = int(n_matches)
                    pnp_inl = int(n_inliers)
                    rmse_b = float(rb)
                    rmse_a = float(ra)

                    print(f"[{frame.Id}] PnP: matches={n_matches} inliers={n_inliers} "
                          f"RMSE(before)={rmse_b:.2f}px RMSE(after)={rmse_a:.2f}px "
                          f"map_pts={len(world_map.points)}")

                    T_world_corr_old = T_world_corr.copy()
                    T_world_corr = T_w_c_pnp @ invert_T(frame.Pose)
                    world_delta = T_world_corr @ invert_T(T_world_corr_old)

                    if len(world_map.points) > 0:
                        X = np.vstack([p.point for p in world_map.points])
                        X2 = apply_T_to_points(world_delta, X)
                        for p, v in zip(world_map.points, X2):
                            p.point = v

                    if len(corr_positions) > 0:
                        C = np.vstack(corr_positions)
                        C2 = apply_T_to_points(world_delta, C)
                        corr_positions = [C2[j] for j in range(len(C2))]

                    T_w_c_corr = T_world_corr @ frame.Pose

            # Loop closure detection
            if (frame.Id % loop_every) == 0 and len(keyframes_for_loop) >= 3:
                loop = detect_loop_closure(keyframes_for_loop, frame, K, min_gap=40, min_inliers=220)
                if loop is not None:
                    kf_id, T_c_cur_c_kf, inl = loop
                    loop_kf = int(kf_id)
                    loop_inl = int(inl)

                    kf = next((k for k in keyframes_for_loop if k.Id == kf_id), None)
                    if kf is not None:
                        T_w_c_kf_corr = T_world_corr @ kf.Pose
                        T_w_c_target = T_w_c_kf_corr @ invert_T(T_c_cur_c_kf)

                        T_world_corr_old = T_world_corr.copy()
                        T_world_corr = T_w_c_target @ invert_T(frame.Pose)
                        world_delta = T_world_corr @ invert_T(T_world_corr_old)

                        if len(world_map.points) > 0:
                            X = np.vstack([p.point for p in world_map.points])
                            X2 = apply_T_to_points(world_delta, X)
                            for p, v in zip(world_map.points, X2):
                                p.point = v

                        if len(corr_positions) > 0:
                            C = np.vstack(corr_positions)
                            C2 = apply_T_to_points(world_delta, C)
                            corr_positions = [C2[j] for j in range(len(C2))]

                        print(f"[{frame.Id}] LOOP CLOSURE with keyframe {kf_id} (inliers={inl}) -> snap corrected")
                        T_w_c_corr = T_world_corr @ frame.Pose

            # Update keyframe occasionally
            if (frame.Id - keyframe.Id) >= keyframe_interval:
                keyframe = frame
                keyframes_for_loop.append(frame)

            # Accumulated trajectory (raw + corrected)
            c_raw = frame.Pose[:3, 3].copy()
            c_corr = T_w_c_corr[:3, 3].copy()

            raw_positions.append(c_raw)
            corr_positions.append(c_corr)

            raw_log.append((frame.Timestamp, float(c_raw[0]), float(c_raw[1]), float(c_raw[2])))
            corr_log.append((frame.Timestamp, float(c_corr[0]), float(c_corr[1]), float(c_corr[2])))

            # 2D XZ trajectory
            traj_img = draw_trajectory_xz(raw_positions, corr_positions, size=700, scale=20.0, center=(350, 350))
            cv2.imshow("trajectory_xz", traj_img)

            # ---- Viewer update: Pangolin must be pumped EVERY frame; rebuild arrays only every show_every ----
            if viewer is not None:
                if (i % show_every) == 0:
                    if len(world_map.points) > 0:
                        viewer_xyz = np.vstack([p.point for p in world_map.points]).astype(np.float64)
                        viewer_rgb = np.vstack([np.array(p.color_bgr, dtype=np.float64) for p in world_map.points]).astype(np.float64)
                    else:
                        viewer_xyz = np.zeros((0, 3), dtype=np.float64)
                        viewer_rgb = np.zeros((0, 3), dtype=np.float64)

                    viewer_traj = np.vstack(corr_positions).astype(np.float64) if len(corr_positions) else np.zeros((0, 3), dtype=np.float64)

                # Pangolin: update every frame (FinishFrame pump). Open3D: keep it lighter.
                if isinstance(viewer, PangolinViewer):
                    viewer.update(viewer_xyz, viewer_rgb, viewer_traj)
                    if viewer.should_quit:
                        break
                else:
                    if (i % show_every) == 0:
                        viewer.update(viewer_xyz, viewer_rgb, viewer_traj)

            writer.writerow({
                "frame": frame.Id,
                "matches_total": m_total,
                "inliers_ransac": m_inl,
                "epi_med_desc": epi_med_desc,
                "epi_mean_desc": epi_mean_desc,
                "epi_med_geom_in": epi_med_geom_in,
                "epi_mean_geom_in": epi_mean_geom_in,
                "pnp_matches": pnp_m,
                "pnp_inliers": pnp_inl,
                "pnp_rmse_before": rmse_b,
                "pnp_rmse_after": rmse_a,
                "loop_kf": loop_kf,
                "loop_inliers": loop_inl,
                "map_points": len(world_map.points)
            })

            prev_frame = frame
            world_map.frames.append(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    finally:
        metrics_f.close()
        if viewer is not None:
            viewer.close()
        cv2.destroyAllWindows()

    # Save with optional recentering to make plots nicer
    if zero_origin_on_save and len(corr_positions) > 0:
        origin = corr_positions[0].copy()
        corr_positions = [p - origin for p in corr_positions]
        corr_log = [(ts, x - float(origin[0]), y - float(origin[1]), z - float(origin[2])) for ts, x, y, z in corr_log]
        for p in world_map.points:
            p.point = p.point - origin

    if len(world_map.points) > 0:
        write_ply(out_ply, world_map.points)
        print(f"Saved map: {out_ply} | points={len(world_map.points)}")

    with open(out_traj_raw, "w", encoding="utf-8") as f:
        for ts, x, y, z in raw_log:
            f.write(f"{ts:.6f} {x:.6f} {y:.6f} {z:.6f}\n")
    print(f"Saved trajectory (raw): {out_traj_raw} (timestamp x y z)")

    with open(out_traj_corr, "w", encoding="utf-8") as f:
        for ts, x, y, z in corr_log:
            f.write(f"{ts:.6f} {x:.6f} {y:.6f} {z:.6f}\n")
    print(f"Saved trajectory (corrected): {out_traj_corr} (timestamp x y z)")

    print(f"Saved metrics: {out_metrics}")


if __name__ == "__main__":
    import argparse

    DEFAULT_DATASET_DIR = r"VO_dataset_SLAM_HW3/rgbd_dataset_freiburg2_pioneer_slam3"

    parser = argparse.ArgumentParser(description="HW3 Monocular VO + Map + PnP + Loop + Metrics")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_DIR, help="Path to dataset folder containing rgb.txt")
    parser.add_argument("--max_frames", type=int, default=600)
    parser.add_argument("--viewer", default="auto", choices=["auto", "pangolin", "open3d", "none"],
                        help="3D viewer backend (HW prefers pangolin).")
    parser.add_argument("--no-matches-window", action="store_true", help="Disable matches_before_after window.")
    parser.add_argument("--no-zero-origin", action="store_true", help="Do not re-center corrected outputs on save.")
    parser.add_argument("--pnp-refine", default="lm", choices=["lm", "gn", "none"],
                        help="PnP optimization method (HW #7): lm (Levenberg-Marquardt), gn (Gauss-Newton), none.")

    args = parser.parse_args()

    main(
        args.dataset,
        max_frames=args.max_frames,
        keyframe_interval=20,
        pnp_every=10,
        loop_every=30,
        show_every=5,
        viewer_mode=args.viewer,
        show_matches_window=(not args.no_matches_window),
        zero_origin_on_save=(not args.no_zero_origin),
        pnp_refine=args.pnp_refine
    )