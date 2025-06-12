import os
import re
import cv2
import pandas as pd

def annotate_pipette_tips(
    excel_path,
    images_folder,
    output_folder,
    time_col_name='Time', 
    tip_col_name_pattern=r'^Tip Coordinates (\d+) \[pix\]$', 
    circle_radius=0, 
    circle_thickness=1,
):
    """
    - excel_path: path to the Excel file containing time stamps and tip coordinates
    - images_folder: folder with input images to be annotated
    - output_folder: folder where annotated images will be saved
    - time_col_name: name of the column in Excel that contains the truncated time stamps 
      (e.g. "17.32" for HH.MM).
    - tip_col_name_pattern: regex pattern to identify columns for pipette X-coordinates.
      By default, it matches "Tip Coordinates 1 [pix]", "Tip Coordinates 2 [pix]", etc.
      The column immediately following each matched column is assumed to be its Y-coordinate.
    - circle_radius: circle radius for marking the pipette tip.
    - circle_thickness: thickness for the circle. -1 means fill the circle.
    - font_scale: scale of the text annotation for labeling each tip.
    - text_thickness: thickness of the annotation text.
    """

    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # 1) Read Excel
    df = pd.read_excel(excel_path)

    # 2) Identify columns that contain X-coordinates for pipettes
    #    We assume the next column after each matched X-coordinate column is its Y-coordinate
    tip_columns = []
    for col in df.columns:
        match = re.match(tip_col_name_pattern, str(col))
        if match:
            pipette_number = match.group(1)
            # The Y-col is immediately after this X-col in the dataframe
            col_index = df.columns.get_loc(col)
            y_col = df.columns[col_index + 1]
            tip_columns.append((col, y_col, pipette_number))

    # 3) Group rows by the truncated time stamp (e.g., "17.32")
    time_groups = {}
    for _, row in df.iterrows():
        truncated_ts = str(row[time_col_name]).strip()
        if truncated_ts not in time_groups:
            time_groups[truncated_ts] = []
        time_groups[truncated_ts].append(row)

    # 4) Parse image file names to get "HH.MM" + last two digits (seconds)
    # Regex captures group(1)=HH, group(2)=MM, group(3)=SS
    image_pattern = re.compile(
        r'^Capture_\d{2}_\d{2}_\d{2}_at_(\d{2})_(\d{2})_(\d{2})(?:\..*)?$',
        re.IGNORECASE
    )

    image_dict = {}
    for fname in sorted(os.listdir(images_folder)):
        match = image_pattern.match(fname)
        if match:
            hh = match.group(1)  # "17"
            mm = match.group(2)  # "32"
            ss = match.group(3)  # "32"
            # FIX the minutes to remove trailing zero  11_20 => "11.2"
            if mm.endswith("0"):
                mm = mm[:-1]  # strip the last character if it's '0'
            truncated_ts = f"{hh}.{mm}"
            last_digits = int(ss)        
            if truncated_ts not in image_dict:
                image_dict[truncated_ts] = []
            image_dict[truncated_ts].append((fname, last_digits))

    # Sort each truncated_ts list by the last two digits (ascending)
    for tts in image_dict:
        image_dict[tts].sort(key=lambda x: x[1])
    # 5) Pair each group of rows in 'time_groups[tts]' with the sorted images in image_dict[tts]
    for tts, rows in time_groups.items():
        # If no corresponding images found for this truncated_ts, skip
        if tts not in image_dict:
            continue

        images_info = image_dict[tts]
        # Only process as many as can be paired
        paired_count = min(len(rows), len(images_info))
        for i in range(paired_count):
            row = rows[i]
            (fname, last_digits) = images_info[i]

            input_image_path = os.path.join(images_folder, fname)
            output_image_path = os.path.join(output_folder, fname)

            # Read the image
            image = cv2.imread(input_image_path)
            if image is None:
                print(f"Warning: could not read {input_image_path}. Skipping.")
                continue

            # Annotate the tips on the image
            for (x_col, y_col, pip_num) in tip_columns:
                x_val = row[x_col]
                y_val = row[y_col]
                if pd.notna(x_val) and pd.notna(y_val):
                    x_coord = int(x_val)
                    y_coord = int(y_val)

                    # Draw a circle
                    cv2.circle(
                        image, 
                        (x_coord, y_coord), 
                        circle_radius, 
                        (255, 0, 0),  # BGR color = red
                        circle_thickness
                    )
                    

            cv2.imwrite(output_image_path, image)
            print(f"Annotated image saved to: {output_image_path}")

if __name__ == "__main__":
    excel_file_path = r"C:\Users\johan\RAWPICS\Tip coordinates 01_09_23_2.xlsx"
    input_images_folder = r"C:\Users\johan\RAWPICS\TOANOT\D23_09_01"
    output_annotated_folder = r"C:\Users\johan\RAWPICS\ANOT"

    annotate_pipette_tips(
        excel_path=excel_file_path,
        images_folder=input_images_folder,
        output_folder=output_annotated_folder
    )
