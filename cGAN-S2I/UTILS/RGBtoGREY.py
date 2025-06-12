import os
import cv2

def convert_images_to_grayscale(source_folder, target_folder, allowed_exts={'.jpg', '.png', '.jpeg', '.bmp', '.tiff'}):
    os.makedirs(target_folder, exist_ok=True)
    
    # Loop over files in source folder
    for filename in os.listdir(source_folder):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in allowed_exts: 
            continue  # Skip non-image files

        src_path = os.path.join(source_folder, filename)
        tgt_path = os.path.join(target_folder, filename)

        # Read image in color
        image = cv2.imread(src_path)
        if image is None:
            print(f"Skipping {filename} (could not read).")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Save to target folder
        success = cv2.imwrite(tgt_path, gray)
        if not success:
            print(f"Failed to save {filename}.")

    print(f"Finished converting images from '{source_folder}' to grayscale in '{target_folder}'.")


convert_images_to_grayscale(r"C:\Users\johan\RAWPICS\ANOT", r"C:\Users\johan\RAWPICS\ANOT\GREY")
