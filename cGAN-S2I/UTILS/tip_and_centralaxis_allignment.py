import os 
import cv2
import numpy as np
import pandas as pd
import re
from sklearn.metrics import pairwise_distances_argmin_min

# Extracts pipette tip coordinates and timestamps from a structured Excel file
def extract_tip_coordinates(excel_path, time_col_name='Time', tip_col_name_pattern=r'^Tip Coordinates (\d+) \[pix\]$'):
    df = pd.read_excel(excel_path)

    # Identify column pairs (X and Y) for tip coordinates based on naming convention
    tip_columns = []
    for col in df.columns:
        match = re.match(tip_col_name_pattern, str(col))
        if match:
            pipette_number = match.group(1)
            col_index = df.columns.get_loc(col)
            y_col = df.columns[col_index + 1]  # Assumes Y coordinate is immediately after X
            tip_columns.append((col, y_col, pipette_number))

    # Group rows by timestamp (used later to align with image timestamps)
    time_groups = {}
    for _, row in df.iterrows():
        truncated_ts = str(row[time_col_name]).strip()
        if truncated_ts not in time_groups:
            time_groups[truncated_ts] = []
        time_groups[truncated_ts].append(row)

    print(f"[DEBUG] Found {len(time_groups)} unique time groups in Excel")
    print(f"[DEBUG] Tip columns: {tip_columns}")
    
    return time_groups, tip_columns

# Parses image filenames to extract the truncated timestamp and order
def parse_image_filenames(images_folder):
    # Matches filenames like: Capture_01_02_03_at_12_34_56_... to extract hours, minutes, seconds
    image_pattern = re.compile(
        r'^Capture_\d{2}_\d{2}_\d{2}_at_(\d{2})_(\d{2})_(\d{2})_.*',
        re.IGNORECASE
    )

    image_dict = {}
    for fname in sorted(os.listdir(images_folder)):
        match = image_pattern.match(fname)
        if match:
            hh = match.group(1)
            mm = match.group(2)
            ss = match.group(3)

            # Truncate minute string if it ends with '0' (matches the Excel convention)
            if mm.endswith("0"):
                mm = mm[:-1]
            truncated_ts = f"{hh}.{mm}"
            last_digits = int(ss)  # used for ordering images within the same truncated timestamp
            if truncated_ts not in image_dict:
                image_dict[truncated_ts] = []
            image_dict[truncated_ts].append((fname, last_digits))

    # Sort each image group by seconds to preserve temporal order
    for tts in image_dict:
        image_dict[tts].sort(key=lambda x: x[1])

    print(f"[DEBUG] Found {sum(len(v) for v in image_dict.values())} total images across {len(image_dict)} truncated timestamps")
    return image_dict

# For each tip coordinate, computes the minimal distance to the nearest point on any line detected in the image
def compute_min_distance_to_lines(binary_image, tip_coords):
    distances = []
    lines = cv2.HoughLinesP(binary_image, 1, np.pi / 180, threshold=30, minLineLength=10, maxLineGap=5)
    if lines is None:
        print("[DEBUG] No lines found by Hough transform")
        return [np.nan] * len(tip_coords)

    # Generate dense set of points along each line segment
    all_points = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        num = max(2, int(np.hypot(x2 - x1, y2 - y1)))  # ensures at least 2 points
        x_vals = np.linspace(x1, x2, num=num)
        y_vals = np.linspace(y1, y2, num=num)
        points = np.column_stack((x_vals, y_vals))
        all_points.append(points)
    all_line_points = np.vstack(all_points)  # combine all points from all lines

    # Compute minimum Euclidean distance from each tip to all line points
    for tip in tip_coords:
        tip_arr = np.array(tip).reshape(1, -1)
        _, dist = pairwise_distances_argmin_min(tip_arr, all_line_points)
        distances.append(dist[0])
    return distances

# Main evaluation routine that pairs Excel tip coordinates with matching image frames
def evaluate_tip_alignment(image_folder, excel_path):
    image_dict = parse_image_filenames(image_folder)
    time_groups, tip_columns = extract_tip_coordinates(excel_path)

    results = []
    for tts, rows in time_groups.items():
        if tts not in image_dict:
            print(f"[DEBUG] No matching images found for time group {tts}")
            continue
        images_info = image_dict[tts]
        paired_count = min(len(rows), len(images_info))  # Align available rows and images
        print(f"[DEBUG] Pairing {paired_count} rows with images for time group {tts}")

        for i in range(paired_count):
            row = rows[i]
            fname, _ = images_info[i]
            image_path = os.path.join(image_folder, fname)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"[WARNING] Could not read image: {image_path}")
                continue

            if np.sum(image) == 0:
                print(f"[WARNING] Image appears to be all black: {fname}")

            # Convert to binary using a fixed threshold
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

            tip_coords = []
            for (x_col, y_col, pip_num) in tip_columns:
                x_val = row[x_col]
                y_val = row[y_col]
                if pd.notna(x_val) and pd.notna(y_val):
                    x_int, y_int = int(x_val), int(y_val)
                    if (x_int, y_int) != (0, 0):  # skip placeholders
                        tip_coords.append((x_int, y_int))
                        print(f"[DEBUG] Found valid tip for image {fname}: ({x_int}, {y_int})")
                    else:
                        print(f"[DEBUG] Skipped placeholder tip (0, 0) for image {fname}")

            if not tip_coords:
                print(f"[DEBUG] No valid tips found for image {fname}")
                continue

            dists = compute_min_distance_to_lines(binary, tip_coords)
            for (tip, dist) in zip(tip_coords, dists):
                results.append({
                    'Image': fname,
                    'Tip_X': tip[0],
                    'Tip_Y': tip[1],
                    'Min_Distance_To_Line': dist
                })

    print(f"[DEBUG] Total results collected: {len(results)}")
    return pd.DataFrame(results)


# Run the alignment evaluation and export results to CSV
df = evaluate_tip_alignment(
    r"C:\Users\dz1921\Downloads\CENTER_AXIS_FROM_MASK_DARK\CENTER_AXIS_FROM_MASK_DARK",
    r"C:\Users\dz1921\Downloads\Tip coordinates 01_09_23_2.xlsx"
)
df.to_csv("output_distances.csv", index=False)