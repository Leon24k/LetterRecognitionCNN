from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import io

app = Flask(__name__)

# Load the trained model
model = load_model("handwritten_alphabet_model.h5")


def preprocess_image(image):
    # Convert image to grayscale
    img = image.convert("L")

    # Resize the image to 28x28 pixels
    img = img.resize((28, 28))

    # Invert colors (optional)
    img = ImageOps.invert(img)

    # Normalize pixel values and reshape to match model input
    img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0

    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    # Load the image from the uploaded file
    image = Image.open(io.BytesIO(file.read()))

    # Preprocess the image
    img_array = preprocess_image(image)

    # Predict the character
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_letter = chr(predicted_class + 65)  # Convert index to ASCII letter (A=65)

    return jsonify({"predicted_letter": predicted_letter})


if __name__ == '__main__':
    app.run(debug=True)
