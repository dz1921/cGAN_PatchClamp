import os
import cv2
import numpy as np

def contrast_stretch_color(image):
    """
    Applies contrast stretching to each channel of a color image (BGR).
    Uses mean ± 2*std range for contrast adjustment.
    """
    # Split into B, G, R channels
    b, g, r = cv2.split(image)

    # Process each channel separately
    b = contrast_stretch(b)
    g = contrast_stretch(g)
    r = contrast_stretch(r)

    # Merge channels back into a color image
    return cv2.merge([b, g, r])

def contrast_stretch(image):
    """
    Applies contrast stretching to an image based on mean ± 2*std.
    Clips pixel values that map outside the range [0,1] and scales them back to [0,255].
    """
    img_float = image.astype(np.float32)

    mean, std_dev = cv2.meanStdDev(img_float)
    mean = mean[0][0]
    std_dev = std_dev[0][0]

    if std_dev < 1e-6:  # Prevent division by zero
        return image

    # Stretching formula
    stretched = (img_float - (mean - 2 * std_dev)) / (4 * std_dev)

    # Clip out-of-range values
    stretched = np.clip(stretched, 0, 1)

    # Scale back to 8-bit range
    return (stretched * 255).astype(np.uint8)

def process_images(source_folder, destination_folder):
    """
    Reads all images from source_folder, applies contrast stretching,
    and writes them to destination_folder while preserving color.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            src_path = os.path.join(source_folder, filename)
            dst_path = os.path.join(destination_folder, filename)

            # Read image in color mode
            image = cv2.imread(src_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Could not open {src_path}. Skipping.")
                continue

            # Apply contrast stretching
            stretched_image = contrast_stretch_color(image)

            # Save processed image
            cv2.imwrite(dst_path, stretched_image)
            print(f"Processed {filename} - {dst_path}")

if __name__ == "__main__":
    source_dir = r"C:\Users\johan\RAWPICS\TOANOT"
    destination_dir = r"C:\Users\johan\RAWPICS\ANOT\CS3"
    
    process_images(source_dir, destination_dir)