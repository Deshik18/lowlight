from flask import Flask, render_template, request, send_file
from test_algo import algorithm_1, algorithm_2, algorithm_3, allowed_file
import os
from PIL import Image
import io

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')  # Main HTML page for uploading images

@app.route('/enhance', methods=['POST'])
def enhance():
    if 'image' not in request.files:
        return "No file part", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    # Check if the file extension is allowed
    if not allowed_file(file.filename):
        return "Invalid file format. Please upload a jpg, jpeg, or png file.", 400

    # Save the uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    # Get the selected algorithm from the form
    selected_algorithm = request.form.get('algorithm')

    # Process the image with the selected algorithm
    if selected_algorithm == 'alg1':
        enhanced_image_path = algorithm_1(image_path)
    elif selected_algorithm == 'alg2':
        enhanced_image_path = algorithm_2(image_path)
    elif selected_algorithm == 'alg3':
        enhanced_image_path = algorithm_3(image_path)
    else:
        return "Invalid algorithm selection", 400

    # Check if the algorithm returned None (e.g., if the image format was wrong)
    if enhanced_image_path is None:
        return "Failed to process the image. Ensure the image format is jpg, jpeg, or png.", 400

    # Open the enhanced image and return it to the user
    enhanced_image = Image.open(enhanced_image_path)
    img_io = io.BytesIO()
    enhanced_image.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run()
