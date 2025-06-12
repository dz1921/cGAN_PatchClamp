import cv2

def resize_with_padding(image, target_size=256, pad_value=0):
    old_size = image.shape[:2]  # (height, width)
    
    # Compute padding sizes
    delta_w = target_size - old_size[1]
    delta_h = target_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # Apply padding
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value)
    
    return padded_image