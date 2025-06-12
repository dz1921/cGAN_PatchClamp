import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from padding_224to256 import resize_with_padding
import cv2

input_dir = "DATA\DIC_IMAGES\TEST_SET"
output_dir = "DATA\DIC_IMAGES\TEST_SET_256"

# Process images
for filename in os.listdir(input_dir):
    if filename.endswith((".png", ".jpg", ".tif")):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        # Resize with zero-padding
        resized_image = resize_with_padding(image, 256)

        cv2.imwrite(output_path, resized_image)

print(f"Resizing complete! Converted images are saved in: {output_dir}")