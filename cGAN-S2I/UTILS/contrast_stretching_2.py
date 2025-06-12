import os
import cv2
import numpy as np

def contrast_stretch_global_color(image):
    """
    Applies contrast stretching to a color image using one global mean and std
    across all channels.
    """
    # Convert to float32
    img_float = image.astype(np.float32)
    
    # Compute a single mean and std over all channels
    # Flatten the array first: shape (rows, cols, 3) -> (rows*cols*3,)
    flat = img_float.reshape(-1)
    mean, std_dev = cv2.meanStdDev(flat)
    mean = mean[0][0]
    std_dev = std_dev[0][0]
    
    if std_dev < 1e-6:  # avoid division by zero
        return image

    # Apply global shift and scale: 
    stretched = (img_float - (mean - 2 * std_dev)) / (4 * std_dev)
    
    # Clip to [0,1]
    stretched = np.clip(stretched, 0, 1)
    
    # Scale to [0,255] and convert back to uint8
    stretched = (stretched * 255).astype(np.uint8)
    
    return stretched

def process_images(source_folder, destination_folder):
    """
    Reads all images from source_folder, applies global contrast stretching in color,
    and writes them to destination_folder.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            src_path = os.path.join(source_folder, filename)
            dst_path = os.path.join(destination_folder, filename)

            image = cv2.imread(src_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Could not open {src_path}. Skipping.")
                continue

            stretched_image = contrast_stretch_global_color(image)
            cv2.imwrite(dst_path, stretched_image)
            print(f"Processed {filename} - {dst_path}")

if __name__ == "__main__":
    source_dir = r"C:\Users\johan\RAWPICS\TOANOT"
    destination_dir = r"C:\Users\johan\RAWPICS\ANOT\MASKS"
    process_images(source_dir, destination_dir)
