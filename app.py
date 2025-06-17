import streamlit as st
import numpy as np
import cv2
from color_checker import auto_detect_patches
from manual_patch_selection import manual_patch_selection
from color_utils import compute_correction_matrix, apply_matrix
from lab_plot import show_input_vs_target_lab_plot  # Only using this plot now

def load_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

st.title("🎨 Modular Color Correction App")

input_img_file = st.file_uploader("Upload input image", type=["jpg", "png"])
target_img_file = st.file_uploader("Upload target image", type=["jpg", "png"])

matrix_input = st.text_area("Enter 3x3 correction matrix", "1,0,0\n0,1,0\n0,0,1")
cct_option = st.selectbox("Select Correlated Color Temperature (CCT) for reference", ["D50", "D65", "D75", "TL84", "Horizon", "Incandescent"])

if input_img_file and target_img_file:
    input_img = load_image(input_img_file)
    target_img = load_image(target_img_file)

    try:
        matrix = np.array([[float(x) for x in row.split(',')] for row in matrix_input.strip().split('\n')])
        assert matrix.shape == (3, 3)
    except:
        st.error("Matrix must be 3x3.")
        st.stop()

    input_patches = auto_detect_patches(input_img)
    target_patches = auto_detect_patches(target_img)

    if input_patches is None or len(input_patches) < 24:
        st.warning("Input image: Auto detection failed, switching to manual mode.")
        input_patches = manual_patch_selection(input_img)

    if target_patches is None or len(target_patches) < 24:
        st.warning("Target image: Auto detection failed, switching to manual mode.")
        target_patches = manual_patch_selection(target_img)

    corrected_matrix = compute_correction_matrix(input_patches, target_patches, matrix)
    corrected_patches = apply_matrix(input_patches, corrected_matrix)

    st.subheader("✅ Refined Correction Matrix")
    st.text(np.array2string(corrected_matrix, precision=4, separator=", "))
    st.download_button("Download Matrix", corrected_matrix.tobytes(), file_name="refined_matrix.npy")

    st.subheader("📊 Input vs Target in a*b* Space")
    show_input_vs_target_lab_plot(input_patches, target_patches)

else:
    st.info("Please upload both an input and target image.")
