import cv2
import numpy as np

def manual_patch_selection(image, patch_rows=4, patch_cols=6):
    print("Manual Rectangular Patch Selection Started...")

    screen_width, screen_height = 1280, 720
    img_height, img_width = image.shape[:2]
    scale = min(screen_width / img_width, screen_height / img_height, 1.0)


    resized_img = cv2.resize(image, (int(img_width * scale), int(img_height * scale)))
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    clone = resized_img.copy()

    selecting = False
    drag_start = None
    current_box = None
    patch_padding = [20]  # % padding

    def draw_preview(img, rect, padding_percent):
        if rect is None:
            return img.copy()
        x1, y1, x2, y2 = rect
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        width = x2 - x1
        height = y2 - y1
        cell_w = width / patch_cols
        cell_h = height / patch_rows

        preview = img.copy()
        pad_x = cell_w * padding_percent / 100 / 2
        pad_y = cell_h * padding_percent / 100 / 2

        # Patch rectangles
        for row in range(patch_rows):
            for col in range(patch_cols):
                px1 = int(x1 + col * cell_w + pad_x)
                py1 = int(y1 + row * cell_h + pad_y)
                px2 = int(x1 + (col + 1) * cell_w - pad_x)
                py2 = int(y1 + (row + 1) * cell_h - pad_y)
                cv2.rectangle(preview, (px1, py1), (px2, py2), (0, 0, 255), 1)

        # Outline of the full chart area
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 1)

        cv2.putText(preview, f"Adjust Padding: {padding_percent}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(preview, f"Drag to reselect box, Enter to confirm", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return preview

    def update_display():
        preview = draw_preview(clone, current_box, patch_padding[0])
        cv2.imshow("Patch Selector", preview)

    def on_mouse(event, x, y, flags, param):
        nonlocal selecting, drag_start, current_box
        if event == cv2.EVENT_LBUTTONDOWN:
            selecting = True
            drag_start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and selecting:
            current_box = drag_start + (x, y)
            update_display()
        elif event == cv2.EVENT_LBUTTONUP:
            selecting = False
            current_box = drag_start + (x, y)
            update_display()

    def on_trackbar(val):
        patch_padding[0] = val
        update_display()

    cv2.namedWindow("Patch Selector")
    cv2.setMouseCallback("Patch Selector", on_mouse)
    cv2.createTrackbar("Padding %", "Patch Selector", patch_padding[0], 50, on_trackbar)

    update_display()
    print("Click and drag to select the Macbeth chart. Adjust padding. Press [Enter] to confirm.")

    while True:
        key = cv2.waitKey(1)
        if key == 13 and current_box:  # Enter
            break
        elif key == 27:  # Esc to cancel
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()

    # Upscale box to original resolution
    rx1, ry1, rx2, ry2 = current_box
    x1 = int(min(rx1, rx2) / scale)
    y1 = int(min(ry1, ry2) / scale)
    x2 = int(max(rx1, rx2) / scale)
    y2 = int(max(ry1, ry2) / scale)

    # Generate patches
    width = x2 - x1
    height = y2 - y1
    cell_w = width / patch_cols
    cell_h = height / patch_rows

    pad_x = cell_w * patch_padding[0] / 100 / 2
    pad_y = cell_h * patch_padding[0] / 100 / 2

    patches = []
    for row in range(patch_rows):
        for col in range(patch_cols):
            px1 = int(x1 + col * cell_w + pad_x)
            py1 = int(y1 + row * cell_h + pad_y)
            px2 = int(x1 + (col + 1) * cell_w - pad_x)
            py2 = int(y1 + (row + 1) * cell_h - pad_y)
            patch = image[py1:py2, px1:px2]
            mean_rgb = np.mean(patch.reshape(-1, 3), axis=0)
            patches.append(mean_rgb)

    return np.array(patches)
