import os
import cv2
import numpy as np


# ----------------------------
# Dataset helpers (TUM format)
# ----------------------------
def load_tum_list(txt_path):
    """
    Reads TUM rgb.txt style:
    timestamp filename
    (ignores comments starting with #)
    """
    items = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            ts = float(parts[0])
            rel = parts[1]
            items.append((ts, rel))
    return items


def build_k_from_image_size(w, h):
    # Allowed: build intrinsics from image size (no calibration)
    fx = 0.9 * w
    fy = fx
    cx = w / 2.0
    cy = h / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)


# ----------------------------
# Geometry helpers
# ----------------------------
def sampson_error(F, pts1, pts2):
    """
    pts1, pts2: Nx2 pixel points
    returns: N Sampson errors
    """
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])

    Fx1 = (F @ pts1_h.T).T
    Ftx2 = (F.T @ pts2_h.T).T
    x2tFx1 = np.sum(pts2_h * Fx1, axis=1)

    denom = Fx1[:, 0]**2 + Fx1[:, 1]**2 + Ftx2[:, 0]**2 + Ftx2[:, 1]**2
    denom = np.maximum(denom, 1e-12)
    return (x2tFx1**2) / denom


def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


# ----------------------------
# Main VO loop (Step 1)
# ----------------------------
def main(dataset_dir, max_frames=300):
    rgb_txt = os.path.join(dataset_dir, "rgb.txt")
    rgb_dir = os.path.join(dataset_dir, "rgb")

    rgb_list = load_tum_list(rgb_txt)
    if len(rgb_list) < 2:
        raise RuntimeError("rgb.txt has too few entries")

    # ORB + Matcher
    orb = cv2.ORB_create(nfeatures=4000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # World pose of camera (start = identity)
    T_w_c = np.eye(4, dtype=np.float64)
    traj_img = np.zeros((700, 700, 3), dtype=np.uint8)

    prev_img = None
    prev_kp, prev_des = None, None
    K = None

    for i in range(min(max_frames, len(rgb_list))):
        ts, rel = rgb_list[i]

        # Try both formats (some datasets store paths differently)
        candidate1 = os.path.join(dataset_dir, rel)
        candidate2 = os.path.join(rgb_dir, os.path.basename(rel))
        img_path = candidate1 if os.path.exists(candidate1) else candidate2

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        if K is None:
            h, w = img.shape[:2]
            K = build_k_from_image_size(w, h)

        kp, des = orb.detectAndCompute(img, None)

        # Show keypoints
        vis_kp = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("current_frame_keypoints", vis_kp)

        if prev_img is not None and prev_des is not None and des is not None and len(des) > 20 and len(prev_des) > 20:
            # KNN matches + ratio test
            knn = bf.knnMatch(prev_des, des, k=2)
            good = []
            for m, n in knn:
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            # Show matches BEFORE geometry filtering
            match_before = cv2.drawMatches(prev_img, prev_kp, img, kp, good[:200], None, flags=2)
            cv2.imshow("matches_before_filter", match_before)

            if len(good) >= 12:
                pts1 = np.float32([prev_kp[m.queryIdx].pt for m in good])
                pts2 = np.float32([kp[m.trainIdx].pt for m in good])

                # Fundamental matrix (RANSAC)
                F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.999)
                if F is not None and mask is not None:
                    mask = mask.ravel().astype(bool)

                    # Epipolar error before/after
                    err_before = sampson_error(F, pts1, pts2)
                    err_after = err_before[mask] if np.any(mask) else np.array([])

                    if i % 10 == 0:
                        print(f"[{i}] matches: {len(good)} | inliers: {int(mask.sum())} | "
                              f"epi_err mean before: {np.mean(err_before):.3f} | "
                              f"after: {np.mean(err_after) if len(err_after) else np.nan:.3f}")

                    inlier_matches = [m for m, keep in zip(good, mask) if keep]
                    match_after = cv2.drawMatches(prev_img, prev_kp, img, kp, inlier_matches[:200], None, flags=2)
                    cv2.imshow("matches_after_ransac", match_after)

                    # Pose from Essential (up-to-scale)
                    E, _ = cv2.findEssentialMat(pts1[mask], pts2[mask], K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                    if E is not None:
                        _, R, t, _ = cv2.recoverPose(E, pts1[mask], pts2[mask], K)

                        # p2 = R p1 + t
                        T_c2_c1 = make_T(R, t)

                        # Accumulate world pose:
                        # T_w_c2 = T_w_c1 * inv(T_c2_c1)
                        T_w_c = T_w_c @ np.linalg.inv(T_c2_c1)

                        c = T_w_c[:3, 3]  # camera center in world
                        x = int(c[0] * 20 + 350)
                        z = int(c[2] * 20 + 350)

                        if 0 <= x < traj_img.shape[1] and 0 <= z < traj_img.shape[0]:
                            cv2.circle(traj_img, (x, z), 2, (0, 255, 0), -1)

                        cv2.imshow("trajectory_xz", traj_img)

        prev_img = img
        prev_kp, prev_des = kp, des

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    DATASET_DIR = r"VO_dataset_SLAM_HW3/rgbd_dataset_freiburg2_pioneer_slam3"
    main(DATASET_DIR, max_frames=400)
