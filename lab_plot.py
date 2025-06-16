import numpy as np
import cv2
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from color_utils import rgb_to_lab, delta_e_cie2000, compute_color_metrics
import base64
from io import BytesIO
from PIL import Image
from color_references import reference_labs



def rgb_list_to_lab(rgb_list):
    rgb_array = np.array(rgb_list, dtype=np.uint8).reshape((-1, 1, 3))
    lab_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2Lab).reshape((-1, 3))
    lab_array = lab_array.astype(np.float32)
    lab_array[:, 1:] -= 128
    return lab_array

def generate_lab_background(reference_lab):

    import cv2

    # Flatten and extract a and b channels from the reference_lab array
    a_values = reference_lab[:, 1]
    b_values = reference_lab[:, 2]

    # Compute dynamic range with padding
    padding = 10
    a_min, a_max = int(np.floor(a_values.min())) - padding, int(np.ceil(a_values.max())) + padding
    b_min, b_max = int(np.floor(b_values.min())) - padding, int(np.ceil(b_values.max())) + padding

    width = a_max - a_min
    height = b_max - b_min

    # Create meshgrid for a and b
    a = np.linspace(a_min, a_max, width)
    b = np.linspace(b_min, b_max, height)
    a_grid, b_grid = np.meshgrid(a, b)

    # L is fixed at 70
    L = np.full_like(a_grid, 70)

    lab = np.stack((L, a_grid, b_grid), axis=-1).astype(np.float64)

    # Convert to RGB
    rgb = lab2rgb(lab)

    # Convert to 8-bit image (clip first to avoid over/underflow)
    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    # Return image and axis ranges
    return rgb_uint8, (a_min, a_max), (b_min, b_max)

def image_to_base64(img):
    pil_img = Image.fromarray(img)
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def annotate_plot(fig, row, col, metrics):
    text = (
        f"Mean chroma: {metrics['Mean Camera Chroma Saturation (%)']:.1f}%<br>"
        f"ΔC00 corr (mean/max): {metrics['Delta C00 (corrected)']['mean']:.2f} / {metrics['Delta C00 (corrected)']['max']:.2f}<br>"
        f"ΔC00 uncorr (mean/max): {metrics['Delta C00 (uncorrected)']['mean']:.2f} / {metrics['Delta C00 (uncorrected)']['max']:.2f}<br>"
        f"ΔE00 (mean/max): {metrics['Delta E00']['mean']:.2f} / {metrics['Delta E00']['max']:.2f}<br>"
        f"WB Zones: {metrics['White Balance Zones']}"
    )
    fig.add_annotation(text=text, showarrow=False, xref=f'x{(row-1)*2+col}', yref=f'y{(row-1)*2+col}',
                       x=90, y=-90, xanchor='right', yanchor='bottom',
                       font=dict(size=14), row=row, col=col)

def show_lab_plot_grid(input_rgb, corrected_rgb, target_rgb, L_value=70, saturation_factor=0.9, cct='D65'):
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Raw Input vs Target", "Corrected Input vs Target",
        "Ideal vs Input", "Ideal vs Target"
    ))

    lab_bg = generate_lab_background(L_value, saturation_factor)
    bg_uri = "data:image/png;base64," + image_to_base64(lab_bg)

    def add_lab_scatter(fig, row, col, reference_lab, comparison_rgb, marker_symbol):
        #print("TEST1 comparison_lab = ",comparison_rgb)
        comparison_lab = rgb_to_lab(comparison_rgb)
        for i, (lab_r, rgb) in enumerate(zip(reference_lab, comparison_rgb)):
            lab_c = comparison_lab[i]
            fig.add_trace(go.Scatter(
                x=[lab_r[1], lab_c[1]], y=[lab_r[2], lab_c[2]],
                mode='lines+markers',
                marker=dict(color=f'rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})', size=8, symbol=marker_symbol,
                            line=dict(width=1, color='black')),
                line=dict(color='black', width=1),
                showlegend=False
            ), row=row, col=col)
        fig.add_layout_image(
            dict(source=bg_uri, xref="x", yref="y",
                 x=-100, y=120, sizex=200, sizey=220,
                 xanchor="left", yanchor="top", sizing="stretch",
                 layer="below"), row=row, col=col)

    #print("TEST2 ref_lab")
    ref_lab = rgb_to_lab(target_rgb)
    add_lab_scatter(fig, 1, 1, ref_lab, input_rgb, "circle")
    add_lab_scatter(fig, 1, 2, ref_lab, corrected_rgb, "circle")

    ideal_lab = reference_labs[cct]  # ✅ FIXED: use selected CCT
    #print("mnogo zle",cct)
    add_lab_scatter(fig, 2, 1, ideal_lab, input_rgb, "circle")
    add_lab_scatter(fig, 2, 2, ideal_lab, target_rgb, "triangle-up")
    #print("Test1 input_rgb",input_rgb)
    metrics_input = compute_color_metrics(input_rgb, ideal_lab)
    #print("Test2 target_rgb ", target_rgb)
    metrics_target = compute_color_metrics(target_rgb, ideal_lab)
    annotate_plot(fig, 2, 1, metrics_input)
    annotate_plot(fig, 2, 2, metrics_target)

    # 🔴 Overlay selected CCT reference LAB values as red 'x' markers
    ref_cct = reference_labs.get(cct)
    if ref_cct is not None:
        for lab_r in ref_cct:
            for col in [1, 2]:  # Ideal vs Input and Ideal vs Target
                fig.add_trace(go.Scatter(
                    x=[lab_r[1]], y=[lab_r[2]],
                    mode='markers',
                    marker=dict(symbol='x', color='red', size=10, line=dict(width=1)),
                    showlegend=False
                ), row=2, col=col)

    bg_img, a_range, b_range = generate_lab_background(reference_labs)

    fig.update_layout(
        images=[dict(
            source=Image.fromarray(bg_img),
            xref="x", yref="y",
            x=a_range[0], y=b_range[1],
            sizex=a_range[1] - a_range[0],
            sizey=b_range[1] - b_range[0],
            sizing="stretch",
            opacity=0.5,
            layer="below"
        )]
    )

    for i in range(1, 5):
        fig['layout'][f'xaxis{i}'].title = 'a*'
        fig['layout'][f'yaxis{i}'].title = 'b*'
        fig['layout'][f'xaxis{i}'].range = [-100, 100]
        fig['layout'][f'yaxis{i}'].range = [-100, 120]

    from plotly.io import to_image
    st.plotly_chart(fig, use_container_width=True, config={
        "scrollZoom": True,
        "displaylogo": False,
        "dragmode": "zoom",
        "modeBarButtonsToAdd": ["pan2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": "CIELAB_plot",
            "height": 1430,
            "width": 2145,
            "scale": 3
        }
    })

    if st.button("Download High-Res PNG"):
        img_bytes = to_image(fig, format="png", width=2145, height=1430, scale=3)
        st.download_button("Save Plot as PNG", img_bytes, file_name="lab_plot_highres.png")
