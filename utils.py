import numpy as np
import cv2
import streamlit as st

def load_image(uploaded_file):
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def parse_matrix(matrix_input):
    try:
        matrix = np.array([[float(x) for x in row.split(',')] for row in matrix_input.strip().split('\n')])
        assert matrix.shape == (3, 3)
        return matrix
    except:
        st.error("Matrix must be a 3x3 format with comma-separated values.")
        return None
