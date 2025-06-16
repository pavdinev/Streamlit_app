import numpy as np
import cv2
from color_references import reference_labs

def compute_correction_matrix(input_rgb, target_rgb, initial_matrix=None):
    """
    Compute the optimal 3x3 color correction matrix using least squares.
    input_rgb and target_rgb are Nx3 arrays.
    """
    input_rgb = np.array(input_rgb, dtype=np.float32)
    target_rgb = np.array(target_rgb, dtype=np.float32)

    # Optional initial matrix
    if initial_matrix is not None:
        input_rgb = np.dot(input_rgb, initial_matrix.T)

    # Solve for best-fit matrix: A * M = B
    A = input_rgb
    B = target_rgb
    M, _, _, _ = np.linalg.lstsq(A, B, rcond=None)  # M is 3x3

    return M.T  # Transpose to match usage: corrected = input @ M.T


def apply_matrix(rgb_data, matrix):
    """
    Apply a 3x3 color correction matrix to a list of RGB values.
    """
    rgb = np.array(rgb_data, dtype=np.float32)
    corrected = np.dot(rgb, matrix.T)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected

# Convert RGB to LAB using OpenCV

def rgb_to_lab(rgb_list):
    #print("Kaput", rgb_list)
    #print(type(rgb_list))
    rgb_array = np.array(rgb_list, dtype=np.uint8).reshape((-1, 1, 3))
    lab_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2Lab).reshape((-1, 3))
    lab_array = lab_array.astype(np.float32)
    lab_array[:, 1:] -= 128
    return lab_array

# Delta E CIE76 (Euclidean distance)
def delta_e_cie76(lab1, lab2):
    return np.linalg.norm(lab1 - lab2, axis=1)

# Delta E CIEDE2000
def delta_e_cie2000(lab1, lab2):
    L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

    avg_L = (L1 + L2) / 2.0
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    avg_C = (C1 + C2) / 2.0

    G = 0.5 * (1 - np.sqrt((avg_C**7) / (avg_C**7 + 25**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)
    avg_Cp = (C1p + C2p) / 2.0

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    delta_hp = h2p - h1p
    delta_hp = np.where(np.abs(delta_hp) > 180, delta_hp - 360 * np.sign(delta_hp), delta_hp)
    delta_Hp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(delta_hp) / 2)

    delta_Lp = L2 - L1
    delta_Cp = C2p - C1p
    avg_Hp = (h1p + h2p + np.where(np.abs(h1p - h2p) > 180, 360, 0)) / 2

    T = (1
         - 0.17 * np.cos(np.radians(avg_Hp - 30))
         + 0.24 * np.cos(np.radians(2 * avg_Hp))
         + 0.32 * np.cos(np.radians(3 * avg_Hp + 6))
         - 0.20 * np.cos(np.radians(4 * avg_Hp - 63)))

    SL = 1 + (0.015 * (avg_L - 50)**2) / np.sqrt(20 + (avg_L - 50)**2)
    SC = 1 + 0.045 * avg_Cp
    SH = 1 + 0.015 * avg_Cp * T

    delta_ro = 30 * np.exp(-((avg_Hp - 275) / 25)**2)
    RC = 2 * np.sqrt((avg_Cp**7) / (avg_Cp**7 + 25**7))
    RT = -RC * np.sin(np.radians(2 * delta_ro))

    delta_E = np.sqrt(
        (delta_Lp / SL)**2 +
        (delta_Cp / SC)**2 +
        (delta_Hp / SH)**2 +
        RT * (delta_Cp / SC) * (delta_Hp / SH)
    )
    return delta_E

# Imatest-style color metrics
def compute_color_metrics(measured_rgb, target_rgb):
    measured_lab = rgb_to_lab(measured_rgb)
    target_lab = rgb_to_lab(target_rgb)
    reference_labs = rgb_to_lab(target_rgb)

    def chroma(lab):
        return np.sqrt(np.square(lab[:, 1]) + np.square(lab[:, 2]))

    chroma_measured = chroma(measured_lab)
    chroma_reference = chroma(reference_labs)

    mean_chroma_measured = np.mean(chroma_measured)
    mean_chroma_reference = np.mean(chroma_reference)
    mean_camera_chroma_saturation = (mean_chroma_measured / mean_chroma_reference) * 100

    delta_c00_uncorrected = np.abs(chroma_measured - chroma_reference)

    corrected_lab = measured_lab.copy()
    corrected_lab[:, 1] = measured_lab[:, 1] * (100 / mean_chroma_measured)
    corrected_lab[:, 2] = measured_lab[:, 2] * (100 / mean_chroma_measured)
    chroma_corrected = chroma(corrected_lab)
    delta_c00_corrected = np.abs(chroma_corrected - chroma_reference)

    delta_e00 = delta_e_cie2000(measured_lab, reference_labs)

    neutral_patch_ids = [19, 20, 21, 22, 23]  # grayscale range
    neutral_dE00 = delta_e00[neutral_patch_ids]

    zone_counts = {
        "Zone 1 (<=1)": int(np.sum(neutral_dE00 <= 1)),
        "Zone 2 (<=2)": int(np.sum((neutral_dE00 > 1) & (neutral_dE00 <= 2))),
        "Zone 3 (<=4)": int(np.sum((neutral_dE00 > 2) & (neutral_dE00 <= 4))),
        "Zone 4 (<=6)": int(np.sum((neutral_dE00 > 4) & (neutral_dE00 <= 6))),
        "Zone 5 (>6)": int(np.sum(neutral_dE00 > 6))
    }

    return {
        "Mean Camera Chroma Saturation (%)": mean_camera_chroma_saturation,
        "Delta C00 (uncorrected)": {
            "mean": float(np.mean(delta_c00_uncorrected)),
            "max": float(np.max(delta_c00_uncorrected))
        },
        "Delta C00 (corrected)": {
            "mean": float(np.mean(delta_c00_corrected)),
            "max": float(np.max(delta_c00_corrected))
        },
        "Delta E00": {
            "mean": float(np.mean(delta_e00)),
            "max": float(np.max(delta_e00))
        },
        "White Balance Zones": zone_counts
    }
