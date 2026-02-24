import os
import sys
import time
import argparse
import glob
import csv
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ----------------------------
# Windows helper: Pangolin DLLs (vcpkg common path)
# ----------------------------
def _win_add_vcpkg_dlls():
    if sys.platform.startswith("win"):
        dll_dir = r"C:\tools\vcpkg\installed\x64-windows\bin"
        try:
            os.add_dll_directory(dll_dir)
        except (FileNotFoundError, OSError):
            pass


_win_add_vcpkg_dlls()
cv2.setUseOptimized(True)


# ============================================================
# REQUIRED: Pose with RotationMatrix + TranslationVector
# ============================================================
@dataclass
class Pose:
    t_world_cam: np.ndarray          # 4x4 camera->world
    rotation_matrix: np.ndarray      # 3x3
    translation_vector: np.ndarray   # 3x1

    @staticmethod
    def from_t(t_world_cam: np.ndarray) -> "Pose":
        t_world_cam = t_world_cam.astype(np.float64)
        r = t_world_cam[:3, :3].copy()
        t = t_world_cam[:3, 3:4].copy()
        return Pose(t_world_cam=t_world_cam, rotation_matrix=r, translation_vector=t)

    def set_t(self, t_world_cam: np.ndarray) -> None:
        self.t_world_cam = t_world_cam.astype(np.float64)
        self.rotation_matrix = self.t_world_cam[:3, :3].copy()
        self.translation_vector = self.t_world_cam[:3, 3:4].copy()


# ============================================================
# REQUIRED: FRAME, Point, Map classes
# ============================================================
@dataclass
class FRAME:
    id: int
    timestamp: float
    pose: Pose
    key_points: list
    descriptors: np.ndarray
    image_gray: np.ndarray
    image_bgr: np.ndarray
    processed: bool = True


@dataclass
class Point:
    id: int
    point: np.ndarray              # (3,) in global/world coords (corrected world)
    frames: list = field(default_factory=list)
    descriptor: Optional[np.ndarray] = None


@dataclass
class Map:
    points: list = field(default_factory=list)
    frames: list = field(default_factory=list)

    _dirty: bool = True
    _desc_cache: Optional[np.ndarray] = None
    _xyz_cache: Optional[np.ndarray] = None

    def mark_dirty(self) -> None:
        self._dirty = True

    def rebuild_cache(self) -> None:
        if not self._dirty:
            return
        if len(self.points) == 0:
            self._desc_cache = np.zeros((0, 32), dtype=np.uint8)
            self._xyz_cache = np.zeros((0, 3), dtype=np.float64)
        else:
            self._desc_cache = np.vstack([p.descriptor for p in self.points]).astype(np.uint8)
            self._xyz_cache = np.vstack([p.point for p in self.points]).astype(np.float64)
        self._dirty = False


# ============================================================
# SE(3) helpers
# ============================================================
def make_t(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r.astype(np.float64)
    out[:3, 3] = t.reshape(3).astype(np.float64)
    return out


def invert_t(t_world_cam: np.ndarray) -> np.ndarray:
    r = t_world_cam[:3, :3]
    t = t_world_cam[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r.T
    out[:3, 3] = -(r.T @ t)
    return out


def rel_t_cam2_cam1(t_world_cam1: np.ndarray, t_world_cam2: np.ndarray) -> np.ndarray:
    # returns T_cam2_cam1 = inv(T_world_cam2) * T_world_cam1
    return invert_t(t_world_cam2) @ t_world_cam1


def apply_t_to_points(t_world_delta: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    if xyz is None or len(xyz) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float64)])
    out_h = (t_world_delta @ xyz_h.T).T
    return out_h[:, :3]


# ============================================================
# Intrinsics from image size (PDF #5)
# ============================================================
def build_k_from_size(w: int, h: int) -> np.ndarray:
    fx = 0.9 * w
    fy = fx
    cx = w / 2.0
    cy = h / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)


# ============================================================
# Dataset loader (Dataset_VO.zip style)
# - If rgb.txt exists, uses it.
# - Else: finds images, sorts by filename.
# ============================================================
def load_image_sequence(dataset_dir: str) -> List[Tuple[float, str]]:
    """
    Robust loader:
    - If dataset_dir contains rgb.txt -> use it.
    - Else, search ONE level down for */rgb.txt (so you can pass the parent folder).
    - If still not found -> fall back to scanning common folders / recursive scan.
    Returns: List[(timestamp, absolute_image_path)]
    """
    import glob
    import os

    # 1) Direct rgb.txt
    rgb_txt = os.path.join(dataset_dir, "rgb.txt")

    # 2) If not found, search one level down (parent folder case)
    if not os.path.exists(rgb_txt):
        one_level = glob.glob(os.path.join(dataset_dir, "*", "rgb.txt"))
        if one_level:
            rgb_txt = one_level[0]
            dataset_dir = os.path.dirname(rgb_txt)

    # If we have rgb.txt, parse it
    if os.path.exists(rgb_txt):
        items = []
        with open(rgb_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                ts, rel = line.split()[:2]
                # Normalize + make safe absolute path
                rel = rel.lstrip("/\\")
                img_path = os.path.normpath(os.path.join(dataset_dir, rel))
                items.append((float(ts), img_path))

        # Optional: filter missing files (prevents crashes if rgb.txt has bad entries)
        items = [(ts, p) for (ts, p) in items if os.path.exists(p)]
        return items

    # -----------------------
    # Fallback: scan folders
    # -----------------------
    candidates = []
    for sub in ["rgb", "images", "img", "frames", "data", "."]:
        d = os.path.join(dataset_dir, sub)
        if os.path.isdir(d):
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
                candidates.extend(glob.glob(os.path.join(d, ext)))

    if len(candidates) == 0:
        # recursive last resort
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            candidates.extend(glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True))

    # filter out depth-like names
    candidates = [p for p in candidates if "depth" not in os.path.basename(p).lower()]
    candidates.sort()

    seq = []
    for i, p in enumerate(candidates):
        seq.append((float(i), os.path.normpath(p)))
    return seq

# ============================================================
# Epipolar error (Sampson distance)
# ============================================================
def sampson_errors(f: Optional[np.ndarray], pts1: np.ndarray, pts2: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if f is None or pts1 is None or pts2 is None or len(pts1) == 0:
        return np.array([], dtype=np.float64)

    x1 = np.hstack([pts1.astype(np.float64), np.ones((len(pts1), 1), dtype=np.float64)])
    x2 = np.hstack([pts2.astype(np.float64), np.ones((len(pts2), 1), dtype=np.float64)])

    fx1 = (f @ x1.T).T
    ftx2 = (f.T @ x2.T).T
    x2tfx1 = np.sum(x2 * fx1, axis=1)

    denom = fx1[:, 0] ** 2 + fx1[:, 1] ** 2 + ftx2[:, 0] ** 2 + ftx2[:, 1] ** 2
    return (x2tfx1 ** 2) / (denom + eps)


def fundamental_8point(pts1: np.ndarray, pts2: np.ndarray) -> Optional[np.ndarray]:
    if pts1 is None or pts2 is None or len(pts1) < 8:
        return None
    f, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    if f is None:
        return None
    if f.shape != (3, 3):
        try:
            f = f.reshape(3, 3)
        except Exception:
            return None
    return f


# ============================================================
# Manual triangulation via DLT (PDF #4: equations)
# ============================================================
def triangulate_dlt(p1: np.ndarray, p2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    # x1,x2 are normalized image coords (N,2) in camera frame (undistorted)
    # p1,p2 are 3x4 projection matrices in normalized coords
    n = len(x1)
    out = np.zeros((n, 3), dtype=np.float64)

    for i in range(n):
        u1, v1 = float(x1[i, 0]), float(x1[i, 1])
        u2, v2 = float(x2[i, 0]), float(x2[i, 1])

        a = np.zeros((4, 4), dtype=np.float64)
        a[0] = u1 * p1[2] - p1[0]
        a[1] = v1 * p1[2] - p1[1]
        a[2] = u2 * p2[2] - p2[0]
        a[3] = v2 * p2[2] - p2[1]

        _, _, vt = np.linalg.svd(a)
        x = vt[-1]
        x = x / max(abs(x[3]), 1e-12)
        out[i] = x[:3]

    return out


# ============================================================
# Reprojection error + pose refinement (LM or GN)
# ============================================================
def project_points(k: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, xw: np.ndarray) -> np.ndarray:
    uv, _ = cv2.projectPoints(xw.astype(np.float64), rvec, tvec, k, None)
    return uv.reshape(-1, 2)


def reprojection_rmse(k: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, xw: np.ndarray, uv: np.ndarray) -> float:
    pred = project_points(k, rvec, tvec, xw)
    err = np.linalg.norm(pred - uv, axis=1)
    return float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan")


def refine_pose_gn_numeric(k: np.ndarray,
                           rvec: np.ndarray,
                           tvec: np.ndarray,
                           xw: np.ndarray,
                           uv: np.ndarray,
                           iters: int = 6,
                           eps: float = 1e-6,
                           damping: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    rvec = rvec.reshape(3, 1).astype(np.float64).copy()
    tvec = tvec.reshape(3, 1).astype(np.float64).copy()

    def residuals(params: np.ndarray) -> np.ndarray:
        rv = params[:3].reshape(3, 1)
        tv = params[3:].reshape(3, 1)
        pred = project_points(k, rv, tv, xw)
        return (uv - pred).reshape(-1)

    params = np.vstack([rvec, tvec]).reshape(-1, 1)

    for _ in range(iters):
        r0 = residuals(params)
        if len(r0) < 12:
            break

        j = np.zeros((len(r0), 6), dtype=np.float64)
        for col in range(6):
            p2 = params.copy()
            p2[col, 0] += eps
            r2 = residuals(p2)
            j[:, col] = (r2 - r0) / eps

        a = j.T @ j + damping * np.eye(6)
        b = -j.T @ r0.reshape(-1, 1)

        try:
            dx = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            break

        params = params + dx

    return params[:3].reshape(3, 1), params[3:].reshape(3, 1)


# ============================================================
# Pangolin viewer (PDF #8: MUST be real-time)
# ============================================================
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

    def update(self,
               points_xyz: np.ndarray,
               traj_raw: np.ndarray,
               traj_corr: np.ndarray,
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

        # Preferred axes: X right, Z forward, Y up -> OpenCV camera y is down, so flip Y for nicer view
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

        # map points (gray)
        if len(pts) > 0:
            gl.glPointSize(2.0)
            gl.glColor3f(0.70, 0.70, 0.70)
            gl.glBegin(gl.GL_POINTS)
            for x, y, z in pts:
                gl.glVertex3f(float(x), float(y), float(z))
            gl.glEnd()

        # raw trajectory (red)
        if len(raw) >= 2:
            gl.glLineWidth(2.0)
            gl.glColor3f(1.0, 0.0, 0.0)
            gl.glBegin(gl.GL_LINE_STRIP)
            for x, y, z in raw:
                gl.glVertex3f(float(x), float(y), float(z))
            gl.glEnd()

        # corrected trajectory (green)
        if len(cor) >= 2:
            gl.glLineWidth(3.0)
            gl.glColor3f(0.0, 1.0, 0.0)
            gl.glBegin(gl.GL_LINE_STRIP)
            for x, y, z in cor:
                gl.glVertex3f(float(x), float(y), float(z))
            gl.glEnd()

        pangolin.FinishFrame()


# ============================================================
# Map building (triangulation keyframe -> current)
# ============================================================
def add_points_from_pair(world_map: Map,
                         keyframe: FRAME,
                         cur_frame: FRAME,
                         t_world_cam_kf_corr: np.ndarray,
                         t_world_cam_cur_corr: np.ndarray,
                         k: np.ndarray,
                         max_new: int,
                         point_id_start: int) -> int:
    if keyframe.descriptors is None or cur_frame.descriptors is None:
        return point_id_start

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(keyframe.descriptors, cur_frame.descriptors, k=2)

    ratio_matches = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            ratio_matches.append(m)

    if len(ratio_matches) < 30:
        return point_id_start

    pts1 = np.float32([keyframe.key_points[m.queryIdx].pt for m in ratio_matches])
    pts2 = np.float32([cur_frame.key_points[m.trainIdx].pt for m in ratio_matches])

    f, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.5, 0.999)
    if f is None or mask is None:
        return point_id_start
    mask = mask.ravel().astype(bool)

    inlier_matches = [m for m, keep in zip(ratio_matches, mask) if keep]
    if len(inlier_matches) < 25:
        return point_id_start

    if len(inlier_matches) > max_new:
        idx = np.random.choice(len(inlier_matches), max_new, replace=False)
        inlier_matches = [inlier_matches[i] for i in idx]

    pts1_in = np.float32([keyframe.key_points[m.queryIdx].pt for m in inlier_matches])
    pts2_in = np.float32([cur_frame.key_points[m.trainIdx].pt for m in inlier_matches])

    # relative pose between cameras (current relative to keyframe)
    t_cam_cur_cam_kf = rel_t_cam2_cam1(t_world_cam_kf_corr, t_world_cam_cur_corr)
    r = t_cam_cur_cam_kf[:3, :3]
    t = t_cam_cur_cam_kf[:3, 3:4]

    # normalized points
    pts1_n = cv2.undistortPoints(pts1_in.reshape(-1, 1, 2), k, None).reshape(-1, 2)
    pts2_n = cv2.undistortPoints(pts2_in.reshape(-1, 1, 2), k, None).reshape(-1, 2)

    p1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    p2 = np.hstack([r, t])

    x_kf = triangulate_dlt(p1, p2, pts1_n, pts2_n)  # points in keyframe cam coords

    # depth check
    z1 = x_kf[:, 2]
    x_cur = (r @ x_kf.T + t).T
    z2 = x_cur[:, 2]
    keep = (z1 > 0.05) & (z2 > 0.05) & (z1 < 400.0) & (z2 < 400.0)

    x_kf = x_kf[keep]
    inlier_matches = [m for m, kkeep in zip(inlier_matches, keep) if kkeep]
    if len(x_kf) < 10:
        return point_id_start

    # transform to world (corrected)
    x_kf_h = np.hstack([x_kf, np.ones((len(x_kf), 1), dtype=np.float64)])
    x_world = (t_world_cam_kf_corr @ x_kf_h.T).T[:, :3]

    added = 0
    for i in range(len(x_world)):
        xyz = x_world[i]
        if not np.isfinite(xyz).all():
            continue
        if np.max(np.abs(xyz)) > 500.0:
            continue

        desc = keyframe.descriptors[inlier_matches[i].queryIdx].copy()
        world_map.points.append(Point(
            id=point_id_start,
            point=xyz.astype(np.float64),
            frames=[keyframe.id, cur_frame.id],
            descriptor=desc
        ))
        point_id_start += 1
        added += 1

    if added > 0:
        world_map.mark_dirty()

    return point_id_start


# ============================================================
# PnP localization + optimization + RMSE
# ============================================================
def localize_pnp(world_map: Map,
                 frame: FRAME,
                 k: np.ndarray,
                 min_corr: int,
                 max_map_match: int,
                 refine_mode: str) -> Optional[Tuple[np.ndarray, int, int, float, float]]:
    if frame.descriptors is None or len(world_map.points) < min_corr:
        return None

    world_map.rebuild_cache()
    map_desc = world_map._desc_cache
    map_xyz = world_map._xyz_cache

    if map_desc is None or map_xyz is None or len(map_desc) < min_corr:
        return None

    # subsample for speed
    if len(map_desc) > max_map_match:
        idx = np.random.choice(len(map_desc), max_map_match, replace=False)
        map_desc = map_desc[idx]
        map_xyz = map_xyz[idx]

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(map_desc, frame.descriptors, k=2)

    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < min_corr:
        return None

    xw = np.float32([map_xyz[m.queryIdx] for m in good])
    uv = np.float32([frame.key_points[m.trainIdx].pt for m in good])

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        xw, uv, k, None,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=6.0,
        confidence=0.999,
        iterationsCount=150
    )
    if (not ok) or (inliers is None) or (len(inliers) < 20):
        return None

    inliers = inliers.reshape(-1)
    xw_in = xw[inliers].astype(np.float64)
    uv_in = uv[inliers].astype(np.float64)

    rmse_before = reprojection_rmse(k, rvec, tvec, xw_in, uv_in)

    rvec_ref, tvec_ref = rvec, tvec
    if refine_mode == "lm":
        if hasattr(cv2, "solvePnPRefineLM"):
            try:
                rvec_ref, tvec_ref = cv2.solvePnPRefineLM(xw_in, uv_in, k, None, rvec, tvec)
            except cv2.error:
                rvec_ref, tvec_ref = rvec, tvec
        else:
            rvec_ref, tvec_ref = refine_pose_gn_numeric(k, rvec, tvec, xw_in, uv_in)
    elif refine_mode == "gn":
        rvec_ref, tvec_ref = refine_pose_gn_numeric(k, rvec, tvec, xw_in, uv_in)

    rmse_after = reprojection_rmse(k, rvec_ref, tvec_ref, xw_in, uv_in)

    r, _ = cv2.Rodrigues(rvec_ref)
    t_cam_world = make_t(r, tvec_ref)
    t_world_cam = invert_t(t_cam_world)

    return t_world_cam, len(good), len(inliers), rmse_before, rmse_after


# ============================================================
# Loop closure detection (throttled; prevents freezing)
# ============================================================
def detect_loop_closure(keyframes: List[FRAME],
                        cur_frame: FRAME,
                        k: np.ndarray,
                        min_gap: int,
                        min_inliers: int,
                        max_candidates: int,
                        time_budget_ms: int) -> Optional[Tuple[int, np.ndarray, int]]:
    if cur_frame.descriptors is None:
        return None

    t0 = time.perf_counter()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    eligible = [kf for kf in keyframes if (cur_frame.id - kf.id) >= min_gap and kf.descriptors is not None]
    if len(eligible) == 0:
        return None

    eligible = eligible[-max_candidates:]

    for kf in reversed(eligible):
        if (time.perf_counter() - t0) * 1000.0 > float(time_budget_ms):
            break

        knn = bf.knnMatch(kf.descriptors, cur_frame.descriptors, k=2)
        good = []
        for pair in knn:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < max(40, min_inliers // 2):
            continue

        pts1 = np.float32([kf.key_points[m.queryIdx].pt for m in good])
        pts2 = np.float32([cur_frame.key_points[m.trainIdx].pt for m in good])

        f, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.5, 0.999)
        if f is None or mask is None:
            continue

        mask = mask.ravel().astype(bool)
        inl = int(mask.sum())
        if inl < min_inliers:
            continue

        pts1_in = pts1[mask]
        pts2_in = pts2[mask]

        e, _ = cv2.findEssentialMat(pts1_in, pts2_in, k, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if e is None:
            continue

        try:
            _, r, t, _ = cv2.recoverPose(e, pts1_in, pts2_in, k)
        except cv2.error:
            continue

        return kf.id, make_t(r, t), inl

    return None


# ============================================================
# OpenCV view composition: ONE required window
# - shows current keypoints
# - shows matches before/after filtering
# - overlays epipolar + pnp reprojection errors
# ============================================================
def compose_status_view(cur_bgr: np.ndarray,
                        keypoints: list,
                        before_matches_img: Optional[np.ndarray],
                        after_matches_img: Optional[np.ndarray],
                        text_lines: List[str],
                        max_width: int = 1280) -> np.ndarray:
    # base keypoints image
    vis_kp = cv2.drawKeypoints(
        cur_bgr, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # resize keypoints view if too wide
    h0, w0 = vis_kp.shape[:2]
    if w0 > max_width:
        scale = max_width / float(w0)
        vis_kp = cv2.resize(vis_kp, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # overlay text
    y = 22
    for line in text_lines[:6]:
        cv2.putText(vis_kp, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y += 22

    # if no matches images, return only keypoints view
    if before_matches_img is None and after_matches_img is None:
        return vis_kp

    # safe defaults
    if before_matches_img is None:
        before_matches_img = np.zeros((240, 320, 3), dtype=np.uint8)
    if after_matches_img is None:
        after_matches_img = np.zeros((240, 320, 3), dtype=np.uint8)

    # downscale matches for speed (keep strip roughly same width as vis_kp)
    def resize_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
        h, w = img.shape[:2]
        if w <= target_w or target_w <= 0:
            return img
        sc = target_w / float(w)
        return cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)

    # Aim: strip width ~= vis_kp width, so each side ~= vis_kp/2
    bw = max(180, vis_kp.shape[1] // 2)

    before_small = resize_to_width(before_matches_img, bw)
    after_small = resize_to_width(after_matches_img, bw)

    # make same height
    hh = max(before_small.shape[0], after_small.shape[0])

    def pad_h(img: np.ndarray, hh_: int) -> np.ndarray:
        if img.shape[0] == hh_:
            return img
        pad = np.zeros((hh_ - img.shape[0], img.shape[1], 3), dtype=np.uint8)
        return np.vstack([img, pad])

    before_small = pad_h(before_small, hh)
    after_small = pad_h(after_small, hh)

    strip = np.hstack([before_small, after_small])

    # label strip
    cv2.putText(strip, "matches BEFORE (ratio only)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(strip, "matches AFTER (RANSAC inliers)", (before_small.shape[1] + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ---- FIX: pad widths so vstack works ----
    def pad_w(img: np.ndarray, ww_: int) -> np.ndarray:
        h, w = img.shape[:2]
        if w == ww_:
            return img
        if w > ww_:
            # if somehow wider, shrink to fit
            sc = ww_ / float(w)
            return cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        pad = np.zeros((h, ww_ - w, 3), dtype=np.uint8)
        return np.hstack([img, pad])

    target_w = max(vis_kp.shape[1], strip.shape[1])
    vis_kp = pad_w(vis_kp, target_w)
    strip = pad_w(strip, target_w)

    out = np.vstack([vis_kp, strip])

    # resize if too wide (final safety)
    if out.shape[1] > max_width:
        sc = max_width / float(out.shape[1])
        out = cv2.resize(out, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)

    return out


# ============================================================
# Main pipeline
# ============================================================
def main(dataset_dir: str,
         max_frames: int,
         resize_scale: float,
         keyframe_interval: int,
         pnp_every: int,
         loop_every: int,
         show_every: int,
         max_new_triangulated: int,
         max_map_match: int,
         refine_mode: str,
         loop_min_gap: int,
         loop_min_inliers: int,
         loop_max_candidates: int,
         loop_budget_ms: int,
         require_pangolin: bool) -> None:

    seq = load_image_sequence(dataset_dir)
    if len(seq) < 2:
        raise RuntimeError("No images found in dataset directory.")

    # outputs
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    out_ply = os.path.join(out_dir, "map_points.ply")
    out_traj_raw = os.path.join(out_dir, "trajectory_raw.txt")
    out_traj_corr = os.path.join(out_dir, "trajectory_corrected.txt")
    out_metrics = os.path.join(out_dir, "metrics.csv")

    metrics_f = open(out_metrics, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(metrics_f, fieldnames=[
        "frame",
        "matches_ratio",
        "inliers_ransac",
        "epi_med_desc",
        "epi_mean_desc",
        "epi_med_geom",
        "epi_mean_geom",
        "pnp_matches",
        "pnp_inliers",
        "pnp_rmse_before",
        "pnp_rmse_after",
        "loop_kf",
        "loop_inliers",
        "map_points"
    ])
    writer.writeheader()

    # Pangolin (required by PDF)
    viewer = PangolinViewer()
    if not viewer.ok and require_pangolin:
        metrics_f.close()
        raise RuntimeError("Pangolin failed to initialize (assignment requires Pangolin real-time).")

    # OpenCV required window (ONE)
    cv2.namedWindow("Current Image + Keypoints", cv2.WINDOW_NORMAL)

    # feature extractor
    orb = cv2.ORB_create(nfeatures=6000)
    # orb = cv2.ORB_create(nfeatures=1500) # tryin with fewer features for speed
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # state
    world_map = Map()
    point_id = 0
    k = None

    t_world_cam_raw = np.eye(4, dtype=np.float64)
    t_world_corr = np.eye(4, dtype=np.float64)

    prev_frame: Optional[FRAME] = None
    keyframe: Optional[FRAME] = None
    keyframes_for_loop: List[FRAME] = []

    raw_positions: List[np.ndarray] = []
    corr_positions: List[np.ndarray] = []
    raw_log: List[Tuple[float, float, float, float]] = []
    corr_log: List[Tuple[float, float, float, float]] = []

    # viewer caches
    pts_cache = np.zeros((0, 3), dtype=np.float64)
    raw_cache = np.zeros((0, 3), dtype=np.float64)
    corr_cache = np.zeros((0, 3), dtype=np.float64)

    try:
        for i in range(min(max_frames, len(seq))):
            ts, img_path = seq[i]

            img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue

            if resize_scale != 1.0:
                img_bgr = cv2.resize(img_bgr, None, fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_AREA)

            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            if k is None:
                h, w = img_gray.shape[:2]
                k = build_k_from_size(w, h)

            kp, des = orb.detectAndCompute(img_gray, None)
            if des is None or len(kp) < 120:
                continue

            frame = FRAME(
                id=i,
                timestamp=ts,
                pose=Pose.from_t(t_world_cam_raw.copy()),
                key_points=kp,
                descriptors=des,
                image_gray=img_gray,
                image_bgr=img_bgr,
                processed=True
            )

            # metrics
            matches_ratio = 0
            inliers_ransac = 0
            epi_med_desc = float("nan")
            epi_mean_desc = float("nan")
            epi_med_geom = float("nan")
            epi_mean_geom = float("nan")

            pnp_matches = 0
            pnp_inliers = 0
            pnp_rmse_before = float("nan")
            pnp_rmse_after = float("nan")

            loop_kf = -1
            loop_inliers = 0

            before_img = None
            after_img = None

            # ----------------------------
            # (1)(2)(4) Motion between consecutive frames via epipolar geometry
            # ----------------------------
            if prev_frame is not None:
                knn = bf.knnMatch(prev_frame.descriptors, frame.descriptors, k=2)

                ratio_matches = []
                for pair in knn:
                    if len(pair) != 2:
                        continue
                    m, n = pair
                    if m.distance < 0.75 * n.distance:
                        ratio_matches.append(m)

                matches_ratio = int(len(ratio_matches))

                if len(ratio_matches) >= 20:
                    pts1 = np.float32([prev_frame.key_points[m.queryIdx].pt for m in ratio_matches])
                    pts2 = np.float32([frame.key_points[m.trainIdx].pt for m in ratio_matches])

                    # epipolar error for descriptor matches (ratio-only) using 8-point
                    f_desc = fundamental_8point(pts1, pts2)
                    se_desc = sampson_errors(f_desc, pts1, pts2)
                    if len(se_desc) > 0:
                        epi_med_desc = float(np.median(se_desc))
                        epi_mean_desc = float(np.mean(se_desc))

                    # geometry with RANSAC
                    f, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.999)
                    if f is not None and mask is not None:
                        mask = mask.ravel().astype(bool)
                        inliers_ransac = int(mask.sum())

                        pts1_in = pts1[mask]
                        pts2_in = pts2[mask]

                        se_geom = sampson_errors(f, pts1_in, pts2_in)
                        if len(se_geom) > 0:
                            epi_med_geom = float(np.median(se_geom))
                            epi_mean_geom = float(np.mean(se_geom))

                        # show matches before/after filtering (required)
                        if (i % show_every) == 0:
                            # BEFORE: ratio-only (not RANSAC)
                            show_ratio = sorted(ratio_matches, key=lambda x: x.distance)[:200]
                            before_img = cv2.drawMatches(
                                prev_frame.image_bgr, prev_frame.key_points,
                                frame.image_bgr, frame.key_points,
                                show_ratio, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                            )

                            # AFTER: RANSAC inliers
                            inlier_matches = [m for m, keep in zip(ratio_matches, mask) if keep]
                            show_inl = sorted(inlier_matches, key=lambda x: x.distance)[:200]
                            after_img = cv2.drawMatches(
                                prev_frame.image_bgr, prev_frame.key_points,
                                frame.image_bgr, frame.key_points,
                                show_inl, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                            )

                        # integrate relative motion from essential matrix
                        if inliers_ransac >= 12:
                            e, _ = cv2.findEssentialMat(pts1_in, pts2_in, k, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                            if e is not None:
                                try:
                                    _, r, t, _ = cv2.recoverPose(e, pts1_in, pts2_in, k)
                                    t = t / max(np.linalg.norm(t), 1e-12)  # direction only
                                    t_cam2_cam1 = make_t(r, t)            # cam1->cam2
                                    t_world_cam_raw = prev_frame.pose.t_world_cam @ invert_t(t_cam2_cam1)
                                    frame.pose.set_t(t_world_cam_raw.copy())
                                except cv2.error:
                                    pass

            # corrected pose from global correction
            t_world_cam_corr = t_world_corr @ frame.pose.t_world_cam

            # init keyframe
            if keyframe is None:
                keyframe = frame
                keyframes_for_loop.append(frame)

            # ----------------------------
            # (4) map: triangulation (manual DLT)
            # ----------------------------
            if keyframe is not None and frame.id != keyframe.id and len(world_map.points) < 15000:
                t_world_cam_kf_corr = t_world_corr @ keyframe.pose.t_world_cam
                point_id = add_points_from_pair(
                    world_map=world_map,
                    keyframe=keyframe,
                    cur_frame=frame,
                    t_world_cam_kf_corr=t_world_cam_kf_corr,
                    t_world_cam_cur_corr=t_world_cam_corr,
                    k=k,
                    max_new=max_new_triangulated,
                    point_id_start=point_id
                )

            # ----------------------------
            # (5)(6)(7) PnP relocalization every N frames + reprojection error + optimization
            # ----------------------------
            if (frame.id % pnp_every) == 0:
                res = localize_pnp(
                    world_map=world_map,
                    frame=frame,
                    k=k,
                    min_corr=60,
                    max_map_match=max_map_match,
                    refine_mode=refine_mode
                )
                if res is not None:
                    t_world_cam_pnp, n_matches, n_inl, rmse_b, rmse_a = res
                    pnp_matches = int(n_matches)
                    pnp_inliers = int(n_inl)
                    pnp_rmse_before = float(rmse_b)
                    pnp_rmse_after = float(rmse_a)

                    print(f"[{frame.id}] PnP: matches={pnp_matches} inliers={pnp_inliers} "
                          f"RMSE(before)={pnp_rmse_before:.2f}px RMSE(after)={pnp_rmse_after:.2f}px "
                          f"map_pts={len(world_map.points)}")

                    # update correction transform and apply to map + corrected traj
                    old = t_world_corr.copy()
                    t_world_corr = t_world_cam_pnp @ invert_t(frame.pose.t_world_cam)
                    delta = t_world_corr @ invert_t(old)

                    # warp map
                    world_map.rebuild_cache()
                    new_xyz = apply_t_to_points(delta, world_map._xyz_cache)
                    for p, v in zip(world_map.points, new_xyz):
                        p.point = v
                    world_map.mark_dirty()

                    # warp corrected past positions
                    if len(corr_positions) > 0:
                        c = np.vstack(corr_positions)
                        c2 = apply_t_to_points(delta, c)
                        corr_positions = [c2[j] for j in range(len(c2))]

                    # update corrected pose
                    t_world_cam_corr = t_world_corr @ frame.pose.t_world_cam

            # ----------------------------
            # (8) Loop closure every M frames (throttled)
            # ----------------------------
            if (frame.id % loop_every) == 0 and len(keyframes_for_loop) >= 3:
                loop = detect_loop_closure(
                    keyframes=keyframes_for_loop,
                    cur_frame=frame,
                    k=k,
                    min_gap=loop_min_gap,
                    min_inliers=loop_min_inliers,
                    max_candidates=loop_max_candidates,
                    time_budget_ms=loop_budget_ms
                )
                if loop is not None:
                    kf_id, t_cam_cur_cam_kf, inl = loop
                    loop_kf = int(kf_id)
                    loop_inliers = int(inl)

                    kf = next((kk for kk in keyframes_for_loop if kk.id == kf_id), None)
                    if kf is not None:
                        # target corrected pose for current frame using loop constraint
                        t_world_cam_kf_corr = t_world_corr @ kf.pose.t_world_cam
                        t_world_cam_target = t_world_cam_kf_corr @ invert_t(t_cam_cur_cam_kf)

                        old = t_world_corr.copy()
                        t_world_corr = t_world_cam_target @ invert_t(frame.pose.t_world_cam)
                        delta = t_world_corr @ invert_t(old)

                        # warp map + corrected trajectory
                        world_map.rebuild_cache()
                        new_xyz = apply_t_to_points(delta, world_map._xyz_cache)
                        for p, v in zip(world_map.points, new_xyz):
                            p.point = v
                        world_map.mark_dirty()

                        if len(corr_positions) > 0:
                            c = np.vstack(corr_positions)
                            c2 = apply_t_to_points(delta, c)
                            corr_positions = [c2[j] for j in range(len(c2))]

                        t_world_cam_corr = t_world_corr @ frame.pose.t_world_cam
                        print(f"[{frame.id}] LOOP CLOSURE with keyframe {kf_id} (inliers={inl}) -> snap corrected")

            # update keyframe
            if keyframe is not None and (frame.id - keyframe.id) >= keyframe_interval:
                keyframe = frame
                keyframes_for_loop.append(frame)
                if len(keyframes_for_loop) > 300:
                    keyframes_for_loop = keyframes_for_loop[-300:]

            # save trajectory
            c_raw = frame.pose.t_world_cam[:3, 3].copy()
            c_corr = t_world_cam_corr[:3, 3].copy()

            raw_positions.append(c_raw)
            corr_positions.append(c_corr)

            raw_log.append((frame.timestamp, float(c_raw[0]), float(c_raw[1]), float(c_raw[2])))
            corr_log.append((frame.timestamp, float(c_corr[0]), float(c_corr[1]), float(c_corr[2])))

            # update pangolin in real-time
            if viewer.ok:
                if (i % show_every) == 0:
                    world_map.rebuild_cache()
                    pts_cache = world_map._xyz_cache if world_map._xyz_cache is not None else np.zeros((0, 3))
                    raw_cache = np.vstack(raw_positions) if len(raw_positions) else np.zeros((0, 3))
                    corr_cache = np.vstack(corr_positions) if len(corr_positions) else np.zeros((0, 3))

                viewer.update(pts_cache, raw_cache, corr_cache, flip_y_for_display=True, max_points_draw=20000)
                if viewer.should_quit:
                    break

            # OpenCV required window (keypoints + matches + numbers)
            text_lines = [
                f"frame={frame.id}  map_pts={len(world_map.points)}  loop(kf,inl)=({loop_kf},{loop_inliers})",
                f"matches_ratio={matches_ratio}  inliers_ransac={inliers_ransac}",
                f"epi(desc) med={epi_med_desc:.3g} mean={epi_mean_desc:.3g} | epi(geom) med={epi_med_geom:.3g} mean={epi_mean_geom:.3g}",
                f"pnp: matches={pnp_matches} inliers={pnp_inliers}  rmse_before={pnp_rmse_before:.2f}px rmse_after={pnp_rmse_after:.2f}px"
            ]
            view = compose_status_view(img_bgr, kp, before_img, after_img, text_lines)
            cv2.imshow("Current Image + Keypoints", view)

            # write metrics
            writer.writerow({
                "frame": frame.id,
                "matches_ratio": matches_ratio,
                "inliers_ransac": inliers_ransac,
                "epi_med_desc": epi_med_desc,
                "epi_mean_desc": epi_mean_desc,
                "epi_med_geom": epi_med_geom,
                "epi_mean_geom": epi_mean_geom,
                "pnp_matches": pnp_matches,
                "pnp_inliers": pnp_inliers,
                "pnp_rmse_before": pnp_rmse_before,
                "pnp_rmse_after": pnp_rmse_after,
                "loop_kf": loop_kf,
                "loop_inliers": loop_inliers,
                "map_points": len(world_map.points)
            })

            prev_frame = frame
            world_map.frames.append(frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    finally:
        metrics_f.close()
        cv2.destroyAllWindows()

    # exports
    # PLY
    if len(world_map.points) > 0:
        with open(out_ply, "w", encoding="utf-8") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(world_map.points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            for p in world_map.points:
                x, y, z = p.point.tolist()
                y = -y  # export with Y-up
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        print(f"Saved map: {out_ply} | points={len(world_map.points)}")

    # trajectories
    with open(out_traj_raw, "w", encoding="utf-8") as f:
        for ts, x, y, z in raw_log:
            y = -y
            f.write(f"{ts:.6f} {x:.6f} {y:.6f} {z:.6f}\n")
    print(f"Saved trajectory (raw): {out_traj_raw}")

    with open(out_traj_corr, "w", encoding="utf-8") as f:
        for ts, x, y, z in corr_log:
            y = -y
            f.write(f"{ts:.6f} {x:.6f} {y:.6f} {z:.6f}\n")
    print(f"Saved trajectory (corrected): {out_traj_corr}")

    print(f"Saved metrics: {out_metrics}")


if __name__ == "__main__":
    # Dataset folder is inside the project folder (same directory as this script)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_DATASET_DIR = os.path.join(SCRIPT_DIR, "VO_dataset_SLAM_HW3", "rgbd_dataset_freiburg2_pioneer_slam3")

    parser = argparse.ArgumentParser(description="HW3 Monocular SLAM (Pangolin required)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_DIR,
                        help="Path to dataset root (default: ./VO_dataset_SLAM_HW3/rgbd_dataset_freiburg2_pioneer_slam3)")
    parser.add_argument("--max_frames", type=int, default=800)
    parser.add_argument("--resize", type=float, default=0.7, help="downscale to speed up")

    # Use hyphenated CLI flags, but map to underscore args.*
    parser.add_argument("--pnp-every", dest="pnp_every", type=int, default=10)
    parser.add_argument("--loop-every", dest="loop_every", type=int, default=60)
    parser.add_argument("--max-loop-candidates", dest="max_loop_candidates", type=int, default=15)
    parser.add_argument("--loop-budget-ms", dest="loop_budget_ms", type=int, default=60)
    parser.add_argument("--max-map-match", dest="max_map_match", type=int, default=5000)

    parser.add_argument("--no-matches-window", dest="no_matches_window", action="store_true")
    parser.add_argument("--pnp-refine", dest="pnp_refine", default="lm", choices=["lm", "gn", "none"])
    parser.add_argument("--no-require-pangolin", dest="no_require_pangolin", action="store_true",
                        help="debug only (won't meet requirement)")
    parser.add_argument("--no-preferred-axes", dest="no_preferred_axes", action="store_true",
                        help="don't flip Y for display/export")

    args = parser.parse_args()

    # Optional: print dataset path once so you can confirm it’s correct
    print(f"[INFO] Using dataset: {args.dataset}")

    main(
        dataset_dir=args.dataset,
        max_frames=args.max_frames,
        resize_scale=args.resize,

        # keep your existing defaults / constants for the rest unless you want to expose them too
        keyframe_interval=20,
        pnp_every=args.pnp_every,
        loop_every=args.loop_every,
        show_every=5,

        max_new_triangulated=500,
        max_map_match=args.max_map_match,
        refine_mode=args.pnp_refine,

        loop_min_gap=40,
        loop_min_inliers=220,
        loop_max_candidates=args.max_loop_candidates,
        loop_budget_ms=args.loop_budget_ms,

        require_pangolin=(not args.no_require_pangolin)
    )