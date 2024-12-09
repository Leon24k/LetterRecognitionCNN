
# Letter Recognition API using Flask and CNN

This project is a Flask-based API that uses a Convolutional Neural Network (CNN) model to recognize letters from image inputs. The model is pre-trained and stored in a model.h5 file. The API provides a single endpoint to upload images and return the predicted letter.




## Features

- Endpoint: /predict
- Accepts image input through a POST request and returns the predicted letter.
- Input: Image file (e.g., JPG, PNG). Output: Predicted letter.



## Installation

### Clone the Repository

```bash
  git clone https://github.com/Leon24k/LetterRecognitionCNN.git

```
```
cd LetterRecognitionCNN
```
### Install Dependencies

Set up a Python virtual environment (optional) and install the required libraries:
```
pip install -r requirements.txt

```
### Add the Model File
Download or copy your model.h5 file into the models/ directory. This file is included in the repository.

### Run the Application
Start the Flask server:
```
flask run
```

 Make a Request to the API
You can test the API using Postman or any HTTP client tool:

- URL: http://127.0.0.1:5000/predict
- Method: POST
- Body: Form-data with key image and value as the image file. 
## Dependencies

The project uses the following main libraries:

- Flask: For building the API.
- TensorFlow/Keras: For loading and using the CNN model.
- Pillow: For handling image processing.


## License

[MIT](https://choosealicense.com/licenses/mit/)

