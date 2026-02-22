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
            parts = line.split()
            if len(parts) >= 2:
                items.append((float(parts[0]), parts[1]))
    return items


def get_image_path(dataset_dir, rel):
    # TUM often stores "rgb/xxxxx.png" in rgb.txt, but sometimes paths differ
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
# Basic SLAM data structures
# ----------------------------
@dataclass
class Frame:
    fid: int
    timestamp: float
    T_w_c: np.ndarray  # 4x4 world-from-camera
    kp: list
    des: np.ndarray
    img_gray: np.ndarray
    img_bgr: np.ndarray


@dataclass
class MapPoint:
    pid: int
    p_w: np.ndarray  # (3,)
    color_bgr: tuple
    obs: list = field(default_factory=list)


# ----------------------------
# Geometry helpers
# ----------------------------
def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def reprojection_errors(P_pix, X_cam, uvs_pix):
    """
    P_pix: 3x4 projection matrix in pixels (e.g., K[I|0])
    X_cam: Nx3 points in that camera coordinate system
    uvs_pix: Nx2 pixel points
    """
    X_h = np.hstack([X_cam, np.ones((len(X_cam), 1))])
    x = (P_pix @ X_h.T).T
    x = x[:, :2] / np.maximum(x[:, 2:3], 1e-12)
    return np.linalg.norm(x - uvs_pix, axis=1)


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


def rel_T_c2_c1(T_w_c1, T_w_c2):
    """
    Given world-from-camera transforms:
      X_w = T_w_c1 * X_c1
      X_w = T_w_c2 * X_c2
    Then:
      X_c2 = inv(T_w_c2) * T_w_c1 * X_c1
    So T_c2_c1 = inv(T_w_c2) @ T_w_c1
    """
    return np.linalg.inv(T_w_c2) @ T_w_c1


# ----------------------------
# Main Step 2 (VO + keyframe triangulation using VO poses)
# ----------------------------
def main(dataset_dir,
         max_frames=600,
         keyframe_interval=20,     # IMPORTANT: increase baseline
         reproj_thresh_px=12.0,    # looser
         min_depth=0.05,
         save_every=50,
         debug_every=10):

    rgb_txt = os.path.join(dataset_dir, "rgb.txt")
    rgb_list = load_tum_list(rgb_txt)
    if len(rgb_list) < 2:
        raise RuntimeError("rgb.txt has too few entries")

    # Output in project folder (same folder as this script)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    out_ply = os.path.join(project_dir, "map_points.ply")

    # Feature + matcher
    orb = cv2.ORB_create(nfeatures=8000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Map storage
    map_points = []
    next_pid = 0

    # Pose accumulation
    T_w_c = np.eye(4, dtype=np.float64)

    prev_frame = None
    keyframe = None
    K = None

    traj_img = np.zeros((700, 700, 3), dtype=np.uint8)

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

        cur_frame = Frame(
            fid=i,
            timestamp=ts,
            T_w_c=T_w_c.copy(),
            kp=kp,
            des=des,
            img_gray=img_gray,
            img_bgr=img_bgr
        )

        # Show keypoints
        vis_kp = cv2.drawKeypoints(img_gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("keypoints", vis_kp)

        # -------------------------------------------------------
        # (A) Visual Odometry: prev -> cur (consecutive)
        # -------------------------------------------------------
        if prev_frame is not None and prev_frame.des is not None:
            knn = bf.knnMatch(prev_frame.des, cur_frame.des, k=2)
            good = []
            for m, n in knn:
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            if len(good) >= 20:
                pts1_all = np.float32([prev_frame.kp[m.queryIdx].pt for m in good])
                pts2_all = np.float32([cur_frame.kp[m.trainIdx].pt for m in good])

                F, fmask = cv2.findFundamentalMat(pts1_all, pts2_all, cv2.FM_RANSAC, 1.0, 0.999)
                if F is not None and fmask is not None:
                    fmask = fmask.ravel().astype(bool)
                    inliers = fmask.sum()

                    if inliers >= 12:
                        pts1 = pts1_all[fmask]
                        pts2 = pts2_all[fmask]

                        # Essential -> pose
                        E, _ = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                        if E is not None:
                            _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

                            # T_c2_c1 maps cam1->cam2
                            T_c2_c1 = make_T(R, t)

                            # Accumulate: T_w_c2 = T_w_c1 * inv(T_c2_c1)
                            T_w_c = prev_frame.T_w_c @ np.linalg.inv(T_c2_c1)
                            cur_frame.T_w_c = T_w_c.copy()

                            # Draw trajectory (x,z)
                            c = T_w_c[:3, 3]
                            x = int(c[0] * 20 + 350)
                            z = int(c[2] * 20 + 350)
                            if 0 <= x < traj_img.shape[1] and 0 <= z < traj_img.shape[0]:
                                cv2.circle(traj_img, (x, z), 2, (0, 255, 0), -1)
                            cv2.imshow("trajectory_xz", traj_img)

        # Initialize keyframe
        if keyframe is None:
            keyframe = cur_frame

        # Decide if we should promote keyframe AFTER using it
        promote_keyframe = (cur_frame.fid - keyframe.fid) >= keyframe_interval

        # -------------------------------------------------------
        # (B) Triangulation: keyframe -> current
        #     Use VO poses to build relative motion (more stable baseline logic)
        # -------------------------------------------------------
        if keyframe is not None and keyframe.fid != cur_frame.fid:
            knn_kf = bf.knnMatch(keyframe.des, cur_frame.des, k=2)
            good_kf = []
            for m, n in knn_kf:
                if m.distance < 0.75 * n.distance:
                    good_kf.append(m)

            if len(good_kf) >= 30:
                pts1_all = np.float32([keyframe.kp[m.queryIdx].pt for m in good_kf])
                pts2_all = np.float32([cur_frame.kp[m.trainIdx].pt for m in good_kf])

                # Quick parallax check (pixel displacement)
                disp = np.median(np.linalg.norm(pts2_all - pts1_all, axis=1))
                if disp < 1.0:
                    # too small baseline
                    pass
                else:
                    # Filter matches with RANSAC F
                    F, fmask = cv2.findFundamentalMat(pts1_all, pts2_all, cv2.FM_RANSAC, 1.5, 0.999)
                    if F is not None and fmask is not None:
                        fmask = fmask.ravel().astype(bool)
                        pts1 = pts1_all[fmask]
                        pts2 = pts2_all[fmask]

                        if len(pts1) >= 20:
                            # Relative transform from keyframe camera to current camera (from VO poses)
                            T_c_cur_c_kf = rel_T_c2_c1(keyframe.T_w_c, cur_frame.T_w_c)
                            R = T_c_cur_c_kf[:3, :3]
                            t = T_c_cur_c_kf[:3, 3:4]

                            # If baseline is tiny in VO scale, skip
                            if np.linalg.norm(t) < 1e-6:
                                pass
                            else:
                                # Normalize image points (no distortion assumed)
                                pts1n = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None).reshape(-1, 2)
                                pts2n = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None).reshape(-1, 2)

                                # Triangulate in keyframe camera coordinates:
                                # P1 = [I|0], P2 = [R|t]
                                P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
                                P2 = np.hstack([R, t])

                                X_h = cv2.triangulatePoints(P1, P2, pts1n.T, pts2n.T).T
                                X_kf = X_h[:, :3] / np.maximum(X_h[:, 3:4], 1e-12)  # in keyframe camera coords

                                if (cur_frame.fid % debug_every) == 0:
                                    print(f"[{cur_frame.fid}] KF matches={len(good_kf)} | F-inliers={len(pts1)} | disp~{disp:.2f}")
                                    print(f"[{cur_frame.fid}] triangulated raw: {len(X_kf)}")

                                # Cheirality (positive depth in both cams)
                                z1 = X_kf[:, 2]
                                X_cur = (R @ X_kf.T + t).T
                                z2 = X_cur[:, 2]
                                keep = (z1 > min_depth) & (z2 > min_depth)

                                X_kf = X_kf[keep]
                                pts1_keep = pts1[keep]
                                pts2_keep = pts2[keep]

                                if (cur_frame.fid % debug_every) == 0:
                                    print(f"[{cur_frame.fid}] after cheirality: {len(X_kf)}")

                                if len(X_kf) > 0:
                                    # Reprojection filtering (pixel space)
                                    P1_pix = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
                                    P2_pix = K @ np.hstack([R, t])

                                    err1 = reprojection_errors(P1_pix, X_kf, pts1_keep)
                                    err2 = reprojection_errors(P2_pix, X_kf, pts2_keep)
                                    keep2 = (err1 < reproj_thresh_px) & (err2 < reproj_thresh_px)

                                    X_kf = X_kf[keep2]
                                    pts1_keep = pts1_keep[keep2]

                                    if (cur_frame.fid % debug_every) == 0:
                                        print(f"[{cur_frame.fid}] after reproj: {len(X_kf)}")

                                    if len(X_kf) > 0:
                                        # Transform to world: X_w = T_w_ckf * X_kf
                                        X_w = (keyframe.T_w_c @ np.hstack([X_kf, np.ones((len(X_kf), 1))]).T).T[:, :3]

                                        # Add to map
                                        for j in range(len(X_w)):
                                            u, v = pts1_keep[j]
                                            ui, vi = int(round(u)), int(round(v))
                                            if 0 <= vi < keyframe.img_bgr.shape[0] and 0 <= ui < keyframe.img_bgr.shape[1]:
                                                color = tuple(int(c) for c in keyframe.img_bgr[vi, ui])
                                            else:
                                                color = (255, 255, 255)

                                            map_points.append(MapPoint(pid=next_pid, p_w=X_w[j].astype(np.float64), color_bgr=color))
                                            next_pid += 1

                                        if (cur_frame.fid % debug_every) == 0:
                                            print(f"[{cur_frame.fid}] total map points: {len(map_points)}")

        # Save occasionally
        if (cur_frame.fid % save_every) == 0 and len(map_points) > 0:
            write_ply(out_ply, map_points)
            print(f"[{cur_frame.fid}] Saved PLY -> {out_ply} | points={len(map_points)}")

        # Promote keyframe at end (IMPORTANT)
        if promote_keyframe:
            keyframe = cur_frame

        prev_frame = cur_frame

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    # Final save
    if len(map_points) > 0:
        write_ply(out_ply, map_points)
        print(f"Final PLY saved -> {out_ply} | points={len(map_points)}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    DATASET_DIR = r"VO_dataset_SLAM_HW3/rgbd_dataset_freiburg2_pioneer_slam3"
    main(
        DATASET_DIR,
        max_frames=600,
        keyframe_interval=20,   # try 20; if still low try 30 or 40
        reproj_thresh_px=12.0,  # try 12; if still low try 20
        min_depth=0.05,
        save_every=50,
        debug_every=10
    )
