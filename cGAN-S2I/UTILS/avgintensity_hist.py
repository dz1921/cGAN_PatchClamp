import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


folder_path = r"C:\Users\johan\RAWPICS\COMBINED_IMAGES"


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

    # Convert image from BGR to RGB (since OpenCV loads images in BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Compute overall average intensity
    overall_intensity = np.mean(image)

    return overall_intensity

def process_folder():
    """
    Processes all images in a folder, computes average intensities, and plots a histogram.
    """
    intensities = []
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    if not image_files:
        print("No images found in the folder!")
        return

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        intensity = calculate_average_intensity(img_path)
        if intensity is not None:
            intensities.append(intensity)

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(intensities, bins=100, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel("Overall Average Intensity")
    plt.ylabel("Number of Images")
    plt.title("Distribution of Image Intensities")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the histogram
    plt.show()

if __name__ == "__main__":
    process_folder()

