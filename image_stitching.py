import cv2
import numpy as np
import argparse

def cylindrical_warp(image: np.ndarray, focal_length: float = None) -> np.ndarray:

    h, w = image.shape[:2]
    if focal_length is None:
        focal_length = w * 1.1

    K = np.array([[focal_length, 0, w / 2],
                  [0, focal_length, h / 2],
                  [0, 0, 1]])

    y_indices, x_indices = np.indices((h, w))

    theta = (x_indices - w / 2) / focal_length
    h_cap = (y_indices - h / 2) / focal_length

    x_prime = focal_length * np.tan(theta) + w / 2
    y_prime = focal_length * (h_cap / np.cos(theta)) + h / 2

    mask = (x_prime >= 0) & (x_prime < w) & (y_prime >= 0) & (y_prime < h)

    warped_image = cv2.remap(image, x_prime.astype(np.float32),
                             y_prime.astype(np.float32), cv2.INTER_LINEAR)
    warped_image[~mask] = 0

    return warped_image

def detect_and_describe(image: np.ndarray, method: str = "sift"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method.lower() == "sift":
        detector = cv2.SIFT_create()
    elif method.lower() == "orb":
        detector = cv2.ORB_create(nfeatures=5000)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features(desc1: np.ndarray, desc2: np.ndarray, method: str = "sift", ratio: float = 0.7):
    if method.lower() == "sift":
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    raw_matches = matcher.knnMatch(desc1, desc2, k=2)
    good = [m for m, n in raw_matches if m.distance < ratio * n.distance]
    return good

def estimate_homography(kp_base, kp_src, matches, reproj_thresh: float = 4.0):

    if len(matches) < 4:
        return None, None

    pts_base = np.float32([kp_base[m.queryIdx].pt for m in matches])
    pts_src = np.float32([kp_src[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(pts_src, pts_base, cv2.RANSAC, reproj_thresh)
    return H, mask


def build_gaussian_pyramid(img, levels):
    gp = [img.astype(np.float32)]
    for _ in range(levels - 1):
        gp.append(cv2.pyrDown(gp[-1]))
    return gp


def build_laplacian_pyramid(img, levels):
    gp = build_gaussian_pyramid(img, levels)
    lp = []
    for i in range(levels - 1):
        size = (gp[i].shape[1], gp[i].shape[0])
        up = cv2.pyrUp(gp[i + 1], dstsize=size)
        lp.append(gp[i] - up)
    lp.append(gp[-1])
    return lp


def multiband_blend(img1, img2, mask, levels=5):
    mask3 = np.dstack([mask] * 3).astype(np.float32)
    lp1 = build_laplacian_pyramid(img1.astype(np.float32), levels)
    lp2 = build_laplacian_pyramid(img2.astype(np.float32), levels)
    gpm = build_gaussian_pyramid(mask3, levels)

    blended_lp = [gm * l1 + (1.0 - gm) * l2 for l1, l2, gm in zip(lp1, lp2, gpm)]

    result = blended_lp[-1]
    for i in range(levels - 2, -1, -1):
        size = (blended_lp[i].shape[1], blended_lp[i].shape[0])
        result = cv2.pyrUp(result, dstsize=size) + blended_lp[i]
    return np.clip(result, 0, 255).astype(np.uint8)

def warp_and_blend(base, src, H, blend=True):
    h_base, w_base = base.shape[:2]
    h_src, w_src = src.shape[:2]

    corners_src = np.float32([[0, 0], [w_src, 0], [w_src, h_src], [0, h_src]]).reshape(-1, 1, 2)
    corners_warped = cv2.perspectiveTransform(corners_src, H)

    corners_all = np.concatenate([
        np.float32([[0, 0], [w_base, 0], [w_base, h_base], [0, h_base]]).reshape(-1, 1, 2),
        corners_warped
    ], axis=0)

    x_min, y_min = corners_all.reshape(-1, 2).min(axis=0).astype(int)
    x_max, y_max = corners_all.reshape(-1, 2).max(axis=0).astype(int)

    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    canvas_w, canvas_h = x_max - x_min, y_max - y_min

    warped_src = cv2.warpPerspective(src, T @ H, (canvas_w, canvas_h))
    canvas_base = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas_base[-y_min:-y_min + h_base, -x_min:-x_min + w_base] = base

    if not blend:
        mask_base = (canvas_base.sum(axis=2) > 0).astype(np.uint8)
        result = warped_src.copy()
        result[mask_base == 1] = canvas_base[mask_base == 1]
        return result

    mask_base_f = (canvas_base.sum(axis=2) > 0).astype(np.float32)
    mask_src_f = (warped_src.sum(axis=2) > 0).astype(np.float32)
    dist_base = cv2.distanceTransform(mask_base_f.astype(np.uint8), cv2.DIST_L2, 5)
    dist_src = cv2.distanceTransform(mask_src_f.astype(np.uint8), cv2.DIST_L2, 5)
    weight_base = dist_base / (dist_base + dist_src + 1e-6)

    return multiband_blend(canvas_base, warped_src, weight_base)

def stitch_images(image_paths, method="sift", blend=True):
    images = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is None: continue
        images.append(cylindrical_warp(img))

    ref_idx = len(images) // 2
    panorama = images[ref_idx]

    for i in range(ref_idx, len(images) - 1):
        kp_base, desc_base = detect_and_describe(panorama, method)
        kp_src, desc_src = detect_and_describe(images[i + 1], method)
        matches = match_features(desc_base, desc_base, method)  # Error fix logic below
        matches = match_features(desc_base, desc_src, method)
        H, _ = estimate_homography(kp_base, kp_src, matches)
        if H is not None:
            panorama = warp_and_blend(panorama, images[i + 1], H, blend)

    for i in range(ref_idx, 0, -1):
        kp_base, desc_base = detect_and_describe(panorama, method)
        kp_src, desc_src = detect_and_describe(images[i - 1], method)
        matches = match_features(desc_base, desc_src, method)
        H, _ = estimate_homography(kp_base, kp_src, matches)
        if H is not None:
            panorama = warp_and_blend(panorama, images[i - 1], H, blend)

    return panorama

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("-o", "--output", default="result_pano.jpg")
    args = parser.parse_args()

    result = stitch_images(args.images)
    cv2.imwrite(args.output, result)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()