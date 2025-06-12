import os
import re
import cv2
import pandas as pd
import numpy as np

# HELPER FUNCTIONS

def morphological_skeleton(binary_image):
    """Perform a basic morphological skeletonization."""
    skel = np.zeros_like(binary_image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    done = False
    current = binary_image.copy()

    while not done:
        eroded = cv2.erode(current, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(current, temp)
        skel = cv2.bitwise_or(skel, temp)
        current = eroded.copy()

        if cv2.countNonZero(current) == 0:
            done = True

    return skel

def fit_lines_and_draw_on_image_x(skel_img, pipette_tips, connectivity=8, line_color=255, thickness=1):
    """
    Fits a single line per connected component of the skeleton
    and extends that line until it leaves the image.
    The line is oriented such that it passes through the
    pipette tip that is closest to each component's centroid.
    """
    def point_to_line_direction(m):
        dx = 1.0 / np.sqrt(1 + m**2)
        dy = m * dx
        return dx, dy

    def extend_to_image_edge(start_x, start_y, dx, dy, img_shape):
        h, w = img_shape
        max_len = max(h, w) * 2
        for i in range(1, max_len):
            x = int(round(start_x + i * dx))
            y = int(round(start_y + i * dy))
            if not (0 <= x < w and 0 <= y < h):
                # Return the last valid point inside the image
                x = int(round(start_x + (i - 1) * dx))
                y = int(round(start_y + (i - 1) * dy))
                return x, y
        return start_x, start_y

    # Ensure skeleton is binary uint8: 0 or 1
    if skel_img.dtype != np.uint8:
        skel_img = skel_img.astype(np.uint8)
    skel_img[skel_img > 0] = 1

    num_labels, labels = cv2.connectedComponents(skel_img, connectivity=connectivity)
    output_img = np.zeros_like(skel_img)

    pipette_tips = np.array(pipette_tips)  # shape: (N,2)

    for label_val in range(1, num_labels):
        component_mask = (labels == label_val)
        coords = np.column_stack(np.where(component_mask))  # (row, col) => (y, x)
        if coords.shape[0] < 2:
            continue

        x_coords = coords[:, 1].astype(np.float32)
        y_coords = coords[:, 0].astype(np.float32)

        # Fit line (y = m*x + b):
        m, _ = np.polyfit(x_coords, y_coords, 1)
        dx, dy = point_to_line_direction(m)

        # Centroid of the component
        centre = np.mean(coords[:, [1, 0]], axis=0)  # (x, y)

        # Find which tip is closest to this component centre
        dists_to_tips = np.linalg.norm(pipette_tips - centre, axis=1)
        closest_tip = pipette_tips[np.argmin(dists_to_tips)]

        # Among the component's pixels, find the one closest to that tip
        dists_to_tip = np.linalg.norm(coords[:, [1, 0]] - closest_tip, axis=1)
        closest_idx = np.argmin(dists_to_tip)
        start_y, start_x = coords[closest_idx]

        # Make sure the direction vector points away from the tip
        v = np.array([dx, dy])
        tip_to_start = np.array([start_x, start_y]) - closest_tip
        if np.dot(v, tip_to_start) < 0:
            dx *= -1
            dy *= -1

        # Extend line to image edge
        end_x, end_y = extend_to_image_edge(start_x, start_y, dx, dy, skel_img.shape)

        cv2.line(output_img, (start_x, start_y), (end_x, end_y), color=line_color, thickness=thickness)

    return output_img

def process_single_image(image_path, tips):
    """
    Applies the canny_edges with the rest of the pipeline to one image,
    using the given pipette tips (list or array of (x, y) coords).
    Returns the final processed mask (uint8).
    """

    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Canny edges
    edges = cv2.Canny(img, 40, 90)

    # Extract and filter contours by length
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    length_threshold = 30
    long_contours = [cnt for cnt in contours if cv2.arcLength(cnt, closed=False) > length_threshold]

    # Draw these contours onto a black image
    black_image = np.zeros_like(img)
    cv2.drawContours(black_image, long_contours, -1, 255, 1)

    # Create skeletonisable objects
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_image = cv2.dilate(black_image, kernel, iterations=1)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=1)

    # Skeletonise
    skeleton = morphological_skeleton(eroded_image)
    skeleton_inv = cv2.bitwise_not(skeleton)

    # Remove edges belonging to the light
    result = cv2.bitwise_and(black_image, skeleton_inv)

    # Re-contour and create final single-pixel lines
    test_image = np.zeros_like(img)
    final_contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(test_image, final_contours, -1, (255), 1)

    test_image_dil = cv2.dilate(test_image, kernel, iterations=1)
    _, binary = cv2.threshold(test_image_dil, 127, 255, cv2.THRESH_BINARY)

    # Sometimes ximgproc works sometimes it does not
    if hasattr(cv2, 'ximgproc'):
        skeletor = cv2.ximgproc.thinning(binary)
    else:
        skeletor = morphological_skeleton(binary)

    # Fit lines and draw them
    output = fit_lines_and_draw_on_image_x(skeletor, tips)

    return output



# MAIN FOLDER-BASED WORKFLOW

def process_all_images_in_folder(
    excel_path,
    images_folder,
    output_folder,
    time_col_name='Time'
):
    """
    1) Reads Excel once; we assume the first column 'Time' might be 13.22, 13.22 etc. for e.g. 13_22_13 or 13_22_55
    2) For each image in 'images_folder':
        - parse the time from the filename
        - find matching row(s) in Excel
        - pick the appropriate row based on seconds in filename
        - extract tip coords
        - run the pipeline (process_single_image)
        - save output to 'output_folder'
    """

    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_excel(excel_path)

    # Gather rows by their truncated "HH.MM"
    # and keep track of repeated times for multiple seconds
    time_groups = {}
    for _, row in df.iterrows():
        if pd.isna(row[time_col_name]):
            continue
        truncated_ts = str(row[time_col_name]).strip().replace(":", ".")
        if truncated_ts not in time_groups:
            time_groups[truncated_ts] = []
        time_groups[truncated_ts].append(row)

    # Identify which columns are X/Y tip coords
    # Look for columns that start with "Tip Coordinates" and end with "[pix]"
    all_cols = list(df.columns)
    x_col_indices = [
        i for i, col in enumerate(all_cols)
        if str(col).startswith("Tip Coordinates") and str(col).endswith("[pix]")
    ]

    # Image file name extraction
    image_pattern = re.compile(
        r'^Capture_\d{2}_\d{2}_\d{2}_at_(\d{2})_(\d{2})_(\d{2})(?:\..*)?$',
        re.IGNORECASE
    )

    # Time stamp to ts
    images_dict = {}
    for fname in sorted(os.listdir(images_folder)):
        #print("Checking file:", fname)
        match = image_pattern.match(fname)
        if match:
            hh = match.group(1)  # e.g. "13"
            mm = match.group(2)  # e.g. "22"
            ss = match.group(3)  # e.g. "13"
            if mm.endswith("0"):
                mm = mm[:-1]
            truncated_ts = f"{hh}.{mm}"  # e.g. "13.22"
            last_digits = int(ss)        # 13
            print(f"Parsed time from filename: {truncated_ts}")
            print(f"Seconds extracted: {ss}")
            if truncated_ts not in images_dict:
                images_dict[truncated_ts] = []
            images_dict[truncated_ts].append((fname, last_digits))
        else:
            print(f"Skipped (no match): {fname}")
    print("\n== Time Keys ==")
    print("From Excel:", sorted(time_groups.keys())[:10])
    print("From Filenames:", sorted(images_dict.keys())[:10])
    common_keys = set(time_groups.keys()).intersection(set(images_dict.keys()))
    print("Matched keys:", sorted(common_keys))
    # Sort each truncated_ts group by seconds ascending
    for tts in images_dict:
        images_dict[tts].sort(key=lambda x: x[1])

    # Pair up each set of rows with the images for that truncated time
    for tts, rows_for_time in time_groups.items():
        if tts not in images_dict:
            continue  # no images for this truncated time

        images_info = images_dict[tts]  # list of (fname, last_digits)
        paired_count = min(len(rows_for_time), len(images_info))

        # Pair them in ascending order
        for i in range(paired_count):
            row = rows_for_time[i]
            (fname, ss) = images_info[i]

            input_path = os.path.join(images_folder, fname)
            output_path = os.path.join(output_folder, fname)

            # Build the tip list from the row
            """
            tip_list = []
            for x_idx in x_col_indices:
                y_idx = x_idx + 1
                if y_idx >= len(all_cols):
                    continue

                x_val = row[all_cols[x_idx]]
                y_val = row[all_cols[y_idx]]
                if pd.notna(x_val) and pd.notna(y_val):
                    tip_list.append((int(round(x_val)), int(round(y_val))))

            tip_list = np.array(tip_list)
            """
            # Build the tip list from the row 2
            tip_list = np.zeros((4,2))
            counter = 0
            for x_idx in x_col_indices:
                y_idx = x_idx + 1
                if y_idx >= len(all_cols):
                    continue

                x_val = row[all_cols[x_idx]]
                y_val = row[all_cols[y_idx]]
                if pd.isna(x_val) or pd.isna(y_val):
                    continue
                try:
                    x_coord = int(round(float(x_val)))
                    y_coord = int(round(float(y_val)))
                    #print("X is: ", x_coord)
                    #print("Y is: ", y_coord)
                    tip_list[counter,0] = x_coord
                    tip_list[counter,1] = y_coord
                    counter += 1
                except:
                    continue

                

            # Process single image
            try:
                result_mask = process_single_image(input_path, tip_list)
            except Exception as e:
                print(f"Skipping {fname} due to error: {e}")
                continue

            # Save result
            cv2.imwrite(output_path, result_mask)
            print(f"Saved processed image to: {output_path}")


if __name__ == "__main__":
    excel_file_path = r"C:\Users\johan\RAWPICS\Tip coordinates 30_08_23.xlsx"
    input_images_folder = r"C:\Users\johan\RAWPICS\TOANOT"
    output_annotated_folder = r"C:\Users\johan\RAWPICS\ANOT\Center_lines"

    process_all_images_in_folder(
        excel_path=excel_file_path,
        images_folder=input_images_folder,
        output_folder=output_annotated_folder,
        time_col_name='Time'
    )
