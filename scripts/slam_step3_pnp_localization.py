import os
import cv2
import numpy as np
from dataclasses import dataclass, field


# ----------------------------
# Dataset helpers (TUM format)
# ----------------------------
def load_tum_list(txt_path):
    items = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ts, rel = line.split()[:2]
            items.append((float(ts), rel))
    return items


def get_image_path(dataset_dir, rel):
    c1 = os.path.join(dataset_dir, rel)
    c2 = os.path.join(dataset_dir, "rgb", os.path.basename(rel))
    return c1 if os.path.exists(c1) else c2


def build_k_from_image_size(w, h):
    fx = 0.9 * w
    fy = fx
    cx = w / 2.0
    cy = h / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class Frame:
    fid: int
    timestamp: float
    T_w_c: np.ndarray
    kp: list
    des: np.ndarray
    img_gray: np.ndarray
    img_bgr: np.ndarray


@dataclass
class MapPoint:
    pid: int
    p_w: np.ndarray        # (3,)
    des: np.ndarray        # (32,) uint8 ORB descriptor
    color_bgr: tuple


# ----------------------------
# SE(3) / projection utilities
# ----------------------------
def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -(R.T @ t)
    return Ti


def rel_T_c2_c1(T_w_c1, T_w_c2):
    # T_c2_c1 = inv(T_w_c2) @ T_w_c1
    return invert_T(T_w_c2) @ T_w_c1


def project_points(K, rvec, tvec, Xw):
    # Xw: Nx3
    uv, _ = cv2.projectPoints(Xw.astype(np.float64), rvec, tvec, K, None)
    return uv.reshape(-1, 2)


def reproj_rmse(K, rvec, tvec, Xw, uv):
    pred = project_points(K, rvec, tvec, Xw)
    err = np.linalg.norm(pred - uv, axis=1)
    return float(np.sqrt(np.mean(err**2))) if len(err) else float("nan")


# ----------------------------
# Pose refinement (Gauss-Newton with finite-diff Jacobian)
# ----------------------------
def refine_pose_gn_numeric(K, rvec, tvec, Xw, uv, iters=7, eps=1e-6, damping=1e-3):
    """
    Minimizes reprojection error using a simple LM-like step:
      (J^T J + damping I) dx = -J^T r
    Jacobian computed by finite differences (easy + stable).
    """
    rvec = rvec.reshape(3, 1).astype(np.float64).copy()
    tvec = tvec.reshape(3, 1).astype(np.float64).copy()

    def residuals(params):
        rv = params[:3].reshape(3, 1)
        tv = params[3:].reshape(3, 1)
        pred = project_points(K, rv, tv, Xw)  # Nx2
        r = (uv - pred).reshape(-1)           # 2N
        return r

    params = np.vstack([rvec, tvec]).reshape(-1, 1)  # 6x1

    for _ in range(iters):
        r0 = residuals(params)  # (2N,)
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
def write_ply(path, points):
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar blue\nproperty uchar green\nproperty uchar red\n")
        f.write("end_header\n")
        for mp in points:
            x, y, z = mp.p_w.tolist()
            b, g, r = mp.color_bgr
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(b)} {int(g)} {int(r)}\n")


# ----------------------------
# Triangulate from keyframe -> current using VO poses
# ----------------------------
def add_points_from_keyframe(map_points, keyframe, cur_frame, K,
                             max_abs_coord=1e4,
                             reproj_thresh_px=12.0,
                             min_depth=0.05,
                             max_new=800,
                             pid_start=0):
    """
    Adds new MapPoints with descriptors taken from keyframe keypoints.
    Returns new pid_start.
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(keyframe.des, cur_frame.des, k=2)

    good = []
    for m, n in knn:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 30:
        return pid_start

    pts1_all = np.float32([keyframe.kp[m.queryIdx].pt for m in good])
    pts2_all = np.float32([cur_frame.kp[m.trainIdx].pt for m in good])

    # RANSAC F
    F, fmask = cv2.findFundamentalMat(pts1_all, pts2_all, cv2.FM_RANSAC, 1.5, 0.999)
    if F is None or fmask is None:
        return pid_start
    fmask = fmask.ravel().astype(bool)

    good = [m for m, keep in zip(good, fmask) if keep]
    if len(good) < 20:
        return pid_start

    # Limit additions
    if len(good) > max_new:
        idx = np.random.choice(len(good), max_new, replace=False)
        good = [good[k] for k in idx]

    pts1 = np.float32([keyframe.kp[m.queryIdx].pt for m in good])
    pts2 = np.float32([cur_frame.kp[m.trainIdx].pt for m in good])

    # Relative pose from VO poses
    T_c_cur_c_kf = rel_T_c2_c1(keyframe.T_w_c, cur_frame.T_w_c)
    R = T_c_cur_c_kf[:3, :3]
    t = T_c_cur_c_kf[:3, 3:4]

    # Normalize points
    pts1n = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    pts2n = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None).reshape(-1, 2)

    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, t])

    X_h = cv2.triangulatePoints(P1, P2, pts1n.T, pts2n.T).T
    X_kf = X_h[:, :3] / np.maximum(X_h[:, 3:4], 1e-12)

    # Cheirality
    z1 = X_kf[:, 2]
    X_cur = (R @ X_kf.T + t).T
    z2 = X_cur[:, 2]
    keep = (z1 > min_depth) & (z2 > min_depth)
    X_kf = X_kf[keep]
    pts1 = pts1[keep]
    pts2 = pts2[keep]
    good = [m for m, k in zip(good, keep) if k]

    if len(X_kf) < 10:
        return pid_start

    # Reprojection filter in pixel space
    P1_pix = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2_pix = K @ np.hstack([R, t])

    def reproj(Ppix, Xcam, uv):
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

    # Transform to world
    X_w = (keyframe.T_w_c @ np.hstack([X_kf, np.ones((len(X_kf), 1))]).T).T[:, :3]

    # Add points with descriptor + color from keyframe
    for j in range(len(X_w)):
        x, y, z = X_w[j]
        if not np.isfinite([x, y, z]).all():
            continue
        if max(abs(x), abs(y), abs(z)) > max_abs_coord:
            continue

        u, v = pts1[j]
        ui, vi = int(round(u)), int(round(v))
        if 0 <= vi < keyframe.img_bgr.shape[0] and 0 <= ui < keyframe.img_bgr.shape[1]:
            color = tuple(int(c) for c in keyframe.img_bgr[vi, ui])
        else:
            color = (255, 255, 255)

        desc = keyframe.des[good[j].queryIdx].copy()
        map_points.append(MapPoint(pid=pid_start, p_w=X_w[j].astype(np.float64), des=desc, color_bgr=color))
        pid_start += 1

    return pid_start


# ----------------------------
# Localization via PnP
# ----------------------------
def localize_pnp(map_points, frame, K, min_corr=50):
    if len(map_points) < min_corr:
        return None

    map_desc = np.vstack([mp.des for mp in map_points])  # Nx32
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(map_desc, frame.des, k=2)

    good = []
    for m, n in knn:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < min_corr:
        return None

    # Build 3D-2D correspondences
    Xw = np.float32([map_points[m.queryIdx].p_w for m in good])
    uv = np.float32([frame.kp[m.trainIdx].pt for m in good])

    # PnP RANSAC
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        Xw, uv, K, None,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=6.0,
        confidence=0.999,
        iterationsCount=200
    )
    if not ok or inliers is None or len(inliers) < 20:
        return None

    inliers = inliers.reshape(-1)
    Xw_in = Xw[inliers].astype(np.float64)
    uv_in = uv[inliers].astype(np.float64)

    rmse_before = reproj_rmse(K, rvec, tvec, Xw_in, uv_in)

    # Refine pose with our GN (numeric jacobian)
    rvec_ref, tvec_ref = refine_pose_gn_numeric(K, rvec, tvec, Xw_in, uv_in, iters=7, damping=1e-3)
    rmse_after = reproj_rmse(K, rvec_ref, tvec_ref, Xw_in, uv_in)

    # Convert to T_w_c
    R, _ = cv2.Rodrigues(rvec_ref)
    T_c_w = make_T(R, tvec_ref)
    T_w_c = invert_T(T_c_w)

    return T_w_c, len(good), len(inliers), rmse_before, rmse_after


# ----------------------------
# Main
# ----------------------------
def main(dataset_dir,
         max_frames=600,
         KEYFRAME_INTERVAL=20,
         PNP_EVERY=10):

    rgb_list = load_tum_list(os.path.join(dataset_dir, "rgb.txt"))
    if len(rgb_list) < 2:
        raise RuntimeError("rgb.txt too short")

    project_dir = os.path.dirname(os.path.abspath(__file__))
    out_ply = os.path.join(project_dir, "map_points.ply")
    out_traj = os.path.join(project_dir, "trajectory.txt")

    orb = cv2.ORB_create(nfeatures=8000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    map_points = []
    pid = 0

    # Pose
    T_w_c = np.eye(4, dtype=np.float64)

    prev_frame = None
    keyframe = None
    K = None

    traj_img = np.zeros((700, 700, 3), dtype=np.uint8)
    traj_log = []

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

        frame = Frame(
            fid=i,
            timestamp=ts,
            T_w_c=T_w_c.copy(),
            kp=kp,
            des=des,
            img_gray=img_gray,
            img_bgr=img_bgr
        )

        # --- VO update ---
        if prev_frame is not None:
            knn = bf.knnMatch(prev_frame.des, frame.des, k=2)
            good = []
            for m, n in knn:
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            if len(good) >= 20:
                pts1_all = np.float32([prev_frame.kp[m.queryIdx].pt for m in good])
                pts2_all = np.float32([frame.kp[m.trainIdx].pt for m in good])

                F, fmask = cv2.findFundamentalMat(pts1_all, pts2_all, cv2.FM_RANSAC, 1.0, 0.999)
                if F is not None and fmask is not None:
                    fmask = fmask.ravel().astype(bool)
                    if fmask.sum() >= 12:
                        pts1 = pts1_all[fmask]
                        pts2 = pts2_all[fmask]

                        E, _ = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                        if E is not None:
                            _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
                            T_c2_c1 = make_T(R, t)
                            T_w_c = prev_frame.T_w_c @ invert_T(T_c2_c1)
                            frame.T_w_c = T_w_c.copy()

        # init keyframe
        if keyframe is None:
            keyframe = frame

        # --- add map points from keyframe ---
        if frame.fid != keyframe.fid:
            pid = add_points_from_keyframe(
                map_points, keyframe, frame, K,
                max_abs_coord=1e4,
                reproj_thresh_px=12.0,
                min_depth=0.05,
                max_new=800,
                pid_start=pid
            )

        # --- PnP localization every N frames ---
        if (frame.fid % PNP_EVERY) == 0 and len(map_points) >= 80:
            res = localize_pnp(map_points, frame, K, min_corr=50)
            if res is not None:
                T_w_c_pnp, n_matches, n_inliers, rmse_b, rmse_a = res
                # Use PnP pose as "localization correction"
                frame.T_w_c = T_w_c_pnp.copy()
                T_w_c = T_w_c_pnp.copy()

                print(f"[{frame.fid}] PnP: matches={n_matches} inliers={n_inliers} "
                      f"RMSE(before)={rmse_b:.2f}px RMSE(after)={rmse_a:.2f}px "
                      f"map_pts={len(map_points)}")

        # --- promote keyframe at interval ---
        if (frame.fid - keyframe.fid) >= KEYFRAME_INTERVAL:
            keyframe = frame

        # trajectory draw/log
        c = frame.T_w_c[:3, 3]
        x = int(c[0] * 20 + 350)
        z = int(c[2] * 20 + 350)
        if 0 <= x < traj_img.shape[1] and 0 <= z < traj_img.shape[0]:
            cv2.circle(traj_img, (x, z), 2, (0, 255, 0), -1)
        cv2.imshow("trajectory_xz", traj_img)

        traj_log.append((frame.timestamp, c[0], c[1], c[2]))

        prev_frame = frame

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    # Save outputs
    if len(map_points) > 0:
        write_ply(out_ply, map_points)
        print(f"Saved map: {out_ply} | points={len(map_points)}")

    with open(out_traj, "w", encoding="utf-8") as f:
        for ts, x, y, z in traj_log:
            f.write(f"{ts:.6f} {x:.6f} {y:.6f} {z:.6f}\n")
    print(f"Saved trajectory: {out_traj} (timestamp x y z)")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    DATASET_DIR = r"VO_dataset_SLAM_HW3/rgbd_dataset_freiburg2_pioneer_slam3"
    main(DATASET_DIR, max_frames=600, KEYFRAME_INTERVAL=20, PNP_EVERY=10)
