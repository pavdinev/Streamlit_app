import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import base64
from skimage.color import rgb2lab  # ✅ Correct color conversion
from color_references import reference_labs


def rgb_to_lab_skimage(rgb_list):
    """
    Converts a list of RGB values (0–255) to CIELAB using skimage.
    """
    rgb_array = np.array(rgb_list, dtype=np.float32) / 255.0  # Normalize
    lab_array = rgb2lab(rgb_array.reshape((-1, 1, 3))).reshape((-1, 3))
    return lab_array


def generate_lab_background(reference_lab, a_range=None, b_range=None):
    import cv2

    # Extract a* and b* values
    a_values = reference_lab[:, 1]
    b_values = reference_lab[:, 2]

    # Compute default dynamic range + padding
    padding = 10
    a_min = int(np.floor(a_values.min())) - padding
    a_max = int(np.ceil(a_values.max())) + padding
    b_min = int(np.floor(b_values.min())) - padding
    b_max = int(np.ceil(b_values.max())) + padding

    # Override if explicit ranges provided
    if a_range is not None:
        a_min, a_max = a_range
    if b_range is not None:
        b_min, b_max = b_range

    width = a_max - a_min
    height = b_max - b_min

    a = np.linspace(a_min, a_max, width)
    b = np.linspace(b_min, b_max, height)
    a_grid, b_grid = np.meshgrid(a, b)

    # Fixed L
    L = np.full_like(a_grid, 70)
    lab = np.stack((L, a_grid, b_grid), axis=-1).astype(np.float64)

    # Convert to RGB using skimage
    from skimage import color
    rgb = color.lab2rgb(lab)

    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return rgb_uint8, (a_min, a_max), (b_min, b_max)


def image_to_base64(img):
    pil_img = Image.fromarray(img)
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def show_input_vs_target_lab_plot(input_rgb, target_rgb, title="Input vs Target (a*b*)"):
    """
    Streamlit-compatible plot: shows LAB a*-b* error vectors between input and target patches,
    overlaid on a CIELAB gamut background (L=70), with custom markers and consistent patch coloring.
    """

    import matplotlib.cm as cm

    input_lab = rgb_to_lab_skimage(input_rgb)
    target_lab = rgb_to_lab_skimage(target_rgb)

    input_a, input_b = input_lab[:, 1], input_lab[:, 2]
    target_a, target_b = target_lab[:, 1], target_lab[:, 2]

    # Get background
    ref_lab = rgb_to_lab_skimage(target_rgb)
    lab_bg, (a_min0, a_max0), (b_min0, b_max0) = generate_lab_background(ref_lab)

    # Compute custom range with +40 padding
    pad = 40
    a_values = np.concatenate([input_a, target_a])
    b_values = np.concatenate([input_b, target_b])

    a_min, a_max = int(np.floor(a_values.min())) - pad, int(np.ceil(a_values.max())) + pad
    b_min, b_max = int(np.floor(b_values.min())) - pad, int(np.ceil(b_values.max())) + pad

    # Generate background with expanded range
    lab_bg, _, _ = generate_lab_background(target_lab, a_range=(a_min, a_max), b_range=(b_min, b_max))

    fig, ax = plt.subplots(figsize=(8, 8))

    # Show LAB gamut background
    ax.imshow(
        lab_bg,
        extent=[a_min, a_max, b_min, b_max],
        origin='lower',
        aspect='auto'
    )

    # Color map for N patches
    n = len(input_lab)
    colors = cm.get_cmap('tab20', n)

    for i in range(n):
        color = colors(i)

        # Draw error arrow
        ax.arrow(input_a[i], input_b[i],
                 target_a[i] - input_a[i],
                 target_b[i] - input_b[i],
                 head_width=1.5, head_length=1.5,
                 fc=color, ec=color, alpha=0.7)

        # Input = circle, Target = triangle
        ax.scatter(input_a[i], input_b[i], color=color, s=80, marker='o', edgecolor='black', zorder=5)
        ax.scatter(target_a[i], target_b[i], color=color, s=80, marker='^', edgecolor='black', zorder=5)

        # Patch index
        ax.text(target_a[i] + 1.5, target_b[i] + 1.5, str(i + 1), fontsize=8, color='black')

    # Add legend manually with proxy artists
    input_proxy = plt.Line2D([0], [0], marker='o', color='w', label='Input',
                             markerfacecolor='red', markersize=8)
    target_proxy = plt.Line2D([0], [0], marker='^', color='w', label='Target',
                              markerfacecolor='green', markersize=8)
    ax.legend(handles=[input_proxy, target_proxy], loc='upper right')

    ax.set_xlim(a_min, a_max)
    ax.set_ylim(b_min, b_max)
    ax.set_xlabel("a*", fontsize=12)
    ax.set_ylabel("b*", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()

    st.pyplot(fig)
