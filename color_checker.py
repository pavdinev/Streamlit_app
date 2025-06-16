import cv2
import numpy as np


def auto_detect_patches(image, debug=False):
    # Resize for faster processing
    orig = image.copy()
    image = cv2.resize(image, (800, int(image.shape[0] * 800 / image.shape[1])))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ORB feature detector
    orb = cv2.ORB_create(1000)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Assume we know the approximate layout of the patches (6 rows x 4 cols)
    # We try to detect a large quadrilateral
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    color_checker_contour = None
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            color_checker_contour = approx
            break

    if color_checker_contour is None:
        print("❌ Auto detection failed.")
        return None

    # Warp the detected region to a flat 6x4 grid
    approx = color_checker_contour.reshape(4, 2)
    approx = sorted(approx, key=lambda p: p[1])  # sort by y
    top = sorted(approx[:2], key=lambda p: p[0])
    bottom = sorted(approx[2:], key=lambda p: p[0])
    ordered = np.array([top[0], top[1], bottom[1], bottom[0]], dtype="float32")

    width, height = 600, 400
    dst = np.array([[0, 0], [width - 1, 0],
                    [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, M, (width, height))

    # Extract 6x4 patch colors
    patches_rgb = []
    h, w = warped.shape[:2]
    dh, dw = h // 6, w // 4
    for row in range(6):
        for col in range(4):
            patch = warped[row * dh:(row + 1) * dh, col * dw:(col + 1) * dw]
            mean_color = cv2.mean(patch)[:3]
            patches_rgb.append(mean_color)
    return np.array(patches_rgb)

