import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SCRIPTS.generate_image import generate_fake_image

def compute_and_visualise_derivative(image_path):
    """
    Loads an image, computes the first derivative using Sobel operator separately 
    for each color channel (R, G, B), and displays the results.

    Args:
        image_path (str): Path to the input image file.
    """
    # Load the image in RGB format
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image '{image_path}'")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from OpenCV BGR to RGB

    # Split channels
    R, G, B = cv2.split(image)

    # Compute first derivatives using Sobel filter (gradient in X and Y directions)
    def compute_derivative(channel):
        grad_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)  # X direction
        grad_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)  # Y direction
        grad = np.sqrt(grad_x**2 + grad_y**2)  # Magnitude of gradient
        grad = np.uint8(255 * grad / np.max(grad))  # Normalise to 0-255
        return grad

    R_derivative = compute_derivative(R)
    G_derivative = compute_derivative(G)
    B_derivative = compute_derivative(B)

    # Merge derivatives into an RGB image
    derivative_image = cv2.merge([R_derivative, G_derivative, B_derivative])

    # Display results
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(R_derivative, cmap="Reds")
    plt.title("Red Channel Derivative")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(G_derivative, cmap="Greens")
    plt.title("Green Channel Derivative")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(B_derivative, cmap="Blues")
    plt.title("Blue Channel Derivative")
    plt.axis("off")

    plt.show()

    # Show combined derivative
    plt.figure(figsize=(5, 5))
    plt.imshow(derivative_image)
    plt.title("Combined RGB Derivative")
    plt.axis("off")
    plt.show()

# Example usage
image_path = r"DATA\ANNOTATED_Pics\annotated_Capture_at_23_07_19_at_17_21_57.png" # Change to your image path
compute_and_visualise_derivative(image_path)
