import cv2
import os


source_folder = r"C:\Users\johan\RAWPICS\TOANOT"
destination_folder = r"C:\Users\johan\RAWPICS\TOANOT\DARK_IMG_200"


if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

def flip_and_save(image_path, save_path, flip_type):
    """
    Loads an image, flips it, and saves it.

    Args:
        image_path (str): Path to input image.
        save_path (str): Path to save flipped image.
        flip_type (int): Flip type (0=Vertical, 1=Horizontal, -1=Both).
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image {image_path}")
        return

    flipped_image = cv2.flip(image, flip_type)
    cv2.imwrite(save_path, flipped_image)

def process_images():
    """
    Goes through all images in the source folder and creates flipped versions.
    """
    image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    if not image_files:
        print("No images found in the folder")
        return

    for img_file in image_files:
        img_path = os.path.join(source_folder, img_file)

        # Save Horizontally Flipped Image
        flip_and_save(img_path, os.path.join(destination_folder, f"hflip_{img_file}"), flip_type=1)

        # Save Vertically Flipped Image
        flip_and_save(img_path, os.path.join(destination_folder, f"vflip_{img_file}"), flip_type=0)

        # Save Both Axes Flipped Image
        flip_and_save(img_path, os.path.join(destination_folder, f"hvflip_{img_file}"), flip_type=-1)

        print(f"Flipped images saved for {img_file}")

if __name__ == "__main__":
    process_images()
    print("Processing complete")
