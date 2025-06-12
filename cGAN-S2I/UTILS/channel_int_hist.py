import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


folder_path = r"C:\Users\johan\RAWPICS\BRIGHT_IMAGES"


def calculate_channel_intensity(image_path):
    """
    Calculates the average intensity for each RGB channel in an image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        tuple: (avg_R, avg_G, avg_B) - Average intensities for Red, Green, and Blue channels.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image {image_path}")
        return None, None, None

    # Convert image from BGR to RGB (since OpenCV loads images in BGR format)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Compute average intensity per channel
    avg_intensity_r = np.mean(image[:, :, 0])  # Red channel
    avg_intensity_g = np.mean(image[:, :, 1])  # Green channel
    avg_intensity_b = np.mean(image[:, :, 2])  # Blue channel

    return avg_intensity_r, avg_intensity_g, avg_intensity_b

def process_folder():
    """
    Processes all images in a folder, computes channel-wise average intensities, and plots histograms.
    """
    intensities_r = []
    intensities_g = []
    intensities_b = []
    
    # Get list of image files
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    if not image_files:
        print("No images found in the folder!")
        return

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        r, g, b = calculate_channel_intensity(img_path)
        if r is not None:
            intensities_r.append(r)
            intensities_g.append(g)
            intensities_b.append(b)

    # Plot histograms for each channel
    plt.figure(figsize=(10, 4))

    #  Red Channel Histogram
    plt.hist(intensities_r, bins=100, color='red', alpha=0.7, edgecolor='black')
    plt.xlabel("Average Intensity (Red Channel)")
    plt.ylabel("Number of Images")
    plt.title("Red Channel Intensity Distribution")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    #  Green Channel Histogram
    plt.figure(figsize=(10, 4))
    plt.hist(intensities_g, bins=100, color='green', alpha=0.7, edgecolor='black')
    plt.xlabel("Average Intensity (Green Channel)")
    plt.ylabel("Number of Images")
    plt.title("Green Channel Intensity Distribution")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    #  Blue Channel Histogram
    plt.figure(figsize=(10, 4))
    plt.hist(intensities_b, bins=100, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel("Average Intensity (Blue Channel)")
    plt.ylabel("Number of Images")
    plt.title("Blue Channel Intensity Distribution")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    process_folder()
