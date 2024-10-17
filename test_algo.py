from PIL import Image
import numpy as np
from algorithms.Backlit.main import process_image  # Correct import
from algorithms.CLIP.test import lowlight  # Correct import
from algorithms.DCE.main import dce_process_image  # Correct import

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Function to check allowed image extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def algorithm_1(image_path):
    if not allowed_file(image_path):
        return None
    enhanced_image_path = process_image(image_path)  # Use the directly imported function
    return enhanced_image_path

def algorithm_2(image_path):
    if not allowed_file(image_path):
        return None
    enhanced_image_path = lowlight(image_path)  # Use the directly imported function
    return enhanced_image_path

def algorithm_3(image_path):
    if not allowed_file(image_path):
        return None
    enhanced_image_path = dce_process_image(image_path)  # Use the directly imported function
    return enhanced_image_path

# print(algorithm_1('tester.jpg'))
# print(algorithm_2('tester.jpg'))
# print(algorithm_3('tester.jpg'))
