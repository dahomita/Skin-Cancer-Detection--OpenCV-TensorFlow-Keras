# Skin Cancer Detection API

This repository contains a FastAPI implementation for skin cancer detection using a pre-trained TensorFlow/Keras model. The API accepts image uploads and returns predictions indicating whether the skin lesion in the image is cancerous or non-cancerous.

## Model Architecture

The system uses a CNN model with the following architecture:

```python
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

## Dataset

The model was trained on the Skin Cancer Binary Classification Dataset from Kaggle: [Skin Cancer Binary Classification Dataset](https://www.kaggle.com/datasets/kylegraupe/skin-cancer-binary-classification-dataset).

## Dependencies

To run the API, install the required dependencies:

```shell
pip install -r requirements.txt
```

Key dependencies include:
- FastAPI
- Uvicorn
- TensorFlow
- OpenCV
- NumPy
- Python-multipart

## API Usage

### Running the API Server

```shell
python main.py
```

This starts the API server on `http://0.0.0.0:8000`.

### API Endpoints

#### POST /predict

Upload an image for skin cancer detection.

**Request:**
- Form data with a file upload named 'file'
- Supported image formats: JPG, JPEG, PNG

**Response:**
JSON object with the following fields:
- `pred_label`: "Cancer" or "Not Cancer"
- `pred_prob`: Probability value between 0 and 1 (rounded to 2 decimal places)

Example response:
```json
{
  "pred_label": "Not Cancer",
  "pred_prob": 0.15
}
```

### Testing the API

You can test the API using tools like cURL, Postman, or with Python:

```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("path/to/skin_image.jpg", "rb")}
response = requests.post(url, files=files)
result = response.json()
print(result)
```

## Deployment

The API can be deployed to various platforms:

1. Cloud services:
   - Heroku
   - AWS (EC2, Elastic Beanstalk)
   - Google Cloud Run
   - Azure App Service

2. Self-hosting:
   - Docker container
   - Virtual private server

## Acknowledgments

- The Skin Cancer Binary Classification Dataset used in this project: [Kaggle](https://www.kaggle.com/datasets/kylegraupe/skin-cancer-binary-classification-dataset).

## License

This project is licensed under the [MIT License](LICENSE).
