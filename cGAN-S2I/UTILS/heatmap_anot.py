import os
import re
import cv2
import numpy as np
import pandas as pd

def create_heatmap(image_size, tip_coords, sigma=4):
    """
    Creates a 1-channel heatmap with Gaussian blobs at tip coordinates.
    - image_size: (height, width)
    - tip_coords: list of (x, y) tuples
    - sigma: standard deviation of the Gaussian
    """
    heatmap = np.zeros(image_size, dtype=np.float32)

    for x, y in tip_coords:
        if 0 <= x < image_size[1] and 0 <= y < image_size[0]:
            # Create meshgrid around the point
            x_grid, y_grid = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0]))
            gaussian = np.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma ** 2))
            heatmap += gaussian
    
    # Clip values to [0, 1]
    heatmap = np.clip(heatmap, 0, 1)
    return (heatmap * 255).astype(np.uint8)


def generate_tip_heatmaps(
    excel_path,
    images_folder,
    output_folder,
    image_size=(256, 256),
    time_col_name='Time',
    tip_col_name_pattern=r'^Tip Coordinates (\d+) \[pix\]$',
    sigma=4
):
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_excel(excel_path)
    
    
    tip_columns = []
    for col in df.columns:
        match = re.match(tip_col_name_pattern, str(col))
        if match:
            col_index = df.columns.get_loc(col)
            y_col = df.columns[col_index + 1]
            tip_columns.append((col, y_col))

    
    time_groups = {}
    for _, row in df.iterrows():
        truncated_ts = str(row[time_col_name]).strip()
        if truncated_ts not in time_groups:
            time_groups[truncated_ts] = []
        time_groups[truncated_ts].append(row)

    image_pattern = re.compile(
        r'^Capture_\d{2}_\d{2}_\d{2}_at_(\d{2})_(\d{2})_(\d{2})(?:\..*)?$',
        re.IGNORECASE
    )

    image_dict = {}
    for fname in sorted(os.listdir(images_folder)):
        match = image_pattern.match(fname)
        if match:
            hh, mm, ss = match.group(1), match.group(2), match.group(3)
            if mm.endswith("0"):
                mm = mm[:-1]
            truncated_ts = f"{hh}.{mm}"
            last_digits = int(ss)
            image_dict.setdefault(truncated_ts, []).append((fname, last_digits))

    for tts in image_dict:
        image_dict[tts].sort(key=lambda x: x[1])

    for tts, rows in time_groups.items():
        if tts not in image_dict:
            continue

        images_info = image_dict[tts]
        paired_count = min(len(rows), len(images_info))

        for i in range(paired_count):
            row = rows[i]
            (fname, _) = images_info[i]
            tip_coords = []

            for (x_col, y_col) in tip_columns:
                x_val = row[x_col]
                y_val = row[y_col]
                if pd.notna(x_val) and pd.notna(y_val):
                    tip_coords.append((int(x_val), int(y_val)))

            heatmap = create_heatmap(image_size, tip_coords, sigma=sigma)
            output_path = os.path.join(output_folder, fname)

            
            cv2.imwrite(output_path, heatmap)
            print(f"Saved heatmap to: {output_path}")

if __name__ == "__main__":
    excel_file_path = r"C:\Users\johan\RAWPICS\Tip coordinates 01_09_23_2.xlsx"
    input_images_folder = r"C:\Users\johan\RAWPICS\TOANOT\D23_09_01"
    output_heatmap_folder = r"C:\Users\johan\RAWPICS\HEATMAPS"

    generate_tip_heatmaps(
        excel_path=excel_file_path,
        images_folder=input_images_folder,
        output_folder=output_heatmap_folder,
        image_size=(256, 256),
        sigma=4
    )