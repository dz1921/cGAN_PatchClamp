import os
import cv2
import numpy as np

def contrast_stretch(image):
    """
    Applies contrast stretching to an image based on
    mean ± 2 * standard deviation. Any pixel values
    that map outside [0, 1] are clipped to 0 or 1.
    
    Returns the stretched image (8-bit range: 0-255).
    """
    # Convert to float32 for calculations
    img_float = image.astype(np.float32)

    # Calculate mean and std
    mean, std_dev = cv2.meanStdDev(img_float)
    mean = mean[0][0]
    std_dev = std_dev[0][0]

    # Avoid division by zero if std_dev is extremely small
    if std_dev < 1e-6:
        return image

    # Map the original pixel values to [0,1] based on [mean - 2*std, mean + 2*std]
    # Formula: new_val = (val - (mean - 2*std)) / (4*std)
    stretched = (img_float - (mean - 2*std_dev)) / (4 * std_dev)

    # Clip any out-of-range values to [0,1]
    stretched = np.clip(stretched, 0, 1)

    # Scale up to [0,255] and convert back to 8-bit
    stretched = (stretched * 255).astype(np.uint8)

    return stretched

def process_images(source_folder, destination_folder):
    """
    Reads all images from source_folder, applies contrast stretching,
    and writes them to destination_folder.
    """
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Loop through each file in the source folder
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            src_path = os.path.join(source_folder, filename)
            dst_path = os.path.join(destination_folder, filename)


            image = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Could not open {src_path}. Skipping.")
                continue

            # Apply the contrast stretching
            stretched_image = contrast_stretch(image)

            # Write the processed file to the destination folder
            cv2.imwrite(dst_path, stretched_image)
            print(f"Processed {filename} - {dst_path}")

if __name__ == "__main__":
    source_dir = r"C:\Users\johan\RAWPICS\TOANOT"
    destination_dir = r"C:\Users\johan\RAWPICS\ANOT\CS1"
    
    process_images(source_dir, destination_dir)
