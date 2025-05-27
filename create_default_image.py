# backend/create_default_image.py

from PIL import Image
import os

def create_default_image():
    # Create a simple 256x256 white image as a placeholder
    img = Image.new('RGB', (256, 256), color='white')
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_image.png")
    img.save(default_path)
    return default_path