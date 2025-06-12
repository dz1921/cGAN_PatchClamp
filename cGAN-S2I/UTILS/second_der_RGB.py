import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_and_visualise_second_derivative(image_path):
    """
    Loads an image, computes the second derivative using the Laplacian operator 
    separately for each color channel (R, G, B), and displays the results.

    Args:
        image_path (str): Path to the input image file.
    """
    # Load the image in RGB format
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image '{image_path}'")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert OpenCV BGR to RGB

    # Split channels
    R, G, B = cv2.split(image)

    # Compute second derivatives using Laplacian operator
    def compute_second_derivative(channel):
        laplacian = cv2.Laplacian(channel, cv2.CV_64F, ksize=3)  # Compute second derivative
        laplacian = np.uint8(255 * np.abs(laplacian) / np.max(np.abs(laplacian)))  # Normalise
        return laplacian

    R_second_derivative = compute_second_derivative(R)
    G_second_derivative = compute_second_derivative(G)
    B_second_derivative = compute_second_derivative(B)

    # Merge derivatives into an RGB image
    second_derivative_image = cv2.merge([R_second_derivative, G_second_derivative, B_second_derivative])

    # Display results
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(R_second_derivative, cmap="Reds")
    plt.title("Red Channel 2nd Derivative")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(G_second_derivative, cmap="Greens")
    plt.title("Green Channel 2nd Derivative")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(B_second_derivative, cmap="Blues")
    plt.title("Blue Channel 2nd Derivative")
    plt.axis("off")

    plt.show()

    # Show combined second derivative image
    plt.figure(figsize=(5, 5))
    plt.imshow(second_derivative_image)
    plt.title("Combined RGB Second Derivative")
    plt.axis("off")
    plt.show()

# Example usage
image_path = r"DATA\ANNOTATED_Pics\annotated_Capture_at_23_07_19_at_17_21_57.png"  # Change to your image path
compute_and_visualise_second_derivative(image_path)
