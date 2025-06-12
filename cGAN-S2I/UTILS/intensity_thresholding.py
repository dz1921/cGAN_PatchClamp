import cv2
import numpy as np
import os
import shutil

# Set the folder path containing images
source_folder = r"C:\Users\johan\RAWPICS\COMBINED_IMAGES"

# Set the folder where filtered images will be copied
destination_folder = r"C:\Users\johan\RAWPICS\DARK_IMAGES"

# Set the intensity threshold 
intensity_threshold = 20

# Ensure destination folder exists
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

def calculate_average_intensity(image_path):
    """
    Calculates the overall average intensity of an RGB image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        float: Overall average intensity.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image {image_path}")
        return None

    # Convert image from BGR to RGB 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Compute overall average intensity
    overall_intensity = np.mean(image)

    return overall_intensity

def process_images():
    """
    Processes all images in the source folder, calculates their intensity, and copies
    images above the threshold to the destination folder.
    """
    image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    if not image_files:
        print("No images found in the folder!")
        return

    for img_file in image_files:
        img_path = os.path.join(source_folder, img_file)
        intensity = calculate_average_intensity(img_path)
        
        if intensity is not None:
            print(f"{img_file} - Intensity: {intensity:.2f}")

            # Copy image if intensity is above the threshold
            if intensity < intensity_threshold:
                shutil.copy(img_path, os.path.join(destination_folder, img_file))
                print(f"Copied {img_file} to {destination_folder}")

if __name__ == "__main__":
    process_images()
    print("Processing complete")
