import numpy as np

def compute_correction_matrix(current_colors, target_colors, initial_matrix):
    corrected = np.dot(current_colors, initial_matrix.T)
    A, _, _, _ = np.linalg.lstsq(corrected, target_colors, rcond=None)
    return np.dot(A.T, initial_matrix)
