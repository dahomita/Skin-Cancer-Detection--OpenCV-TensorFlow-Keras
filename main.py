from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import uvicorn
import io

app = FastAPI(title="Skin Cancer Detection API")

# Load the trained model
model = load_model('skin_cancer_detection_model.h5')

# Define the size of the input images
img_size = (224, 224)

# Define a function to preprocess the input image
def preprocess_image(img):
    img = cv2.resize(img, img_size)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

@app.post("/predict", response_class=JSONResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict if the uploaded skin image contains cancer.
    Returns prediction label and probability.
    """
    # Read the image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Preprocess the image
    processed_img = preprocess_image(img)
    
    # Make prediction
    pred = model.predict(processed_img)
    pred_prob = float(pred[0][0])
    pred_label = "Cancer" if pred_prob > 0.5 else "Not Cancer"
    
    # Return prediction as JSON
    return {
        "pred_label": pred_label,
        "pred_prob": round(pred_prob, 2)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 