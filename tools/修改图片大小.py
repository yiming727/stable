import cv2
import numpy as np
import torch

import torch
from torchvision import transforms
from PIL import Image


def resize_image(image_path, output_path, new_size):
    """
    Resize an image to the specified size.

    Parameters:
    - image_path: str, path to the input image
    - output_path: str, path to save the resized image
    - new_size: tuple, new size of the image (width, height)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Define a transform to resize the image
    resize_transform = transforms.Compose([
        transforms.Resize(new_size),  # Resize to (width, height)
        transforms.ToTensor()  # Convert PIL image to tensor
    ])

    # Apply transform
    image_resized = resize_transform(image)

    # Save resized image
    image_resized_pil = transforms.ToPILImage()(image_resized)
    image_resized_pil.save(output_path)


# Example usage
resize_image('Data/0000011_101010.jpg', './0000011_101010.jpg', (360, 640))

