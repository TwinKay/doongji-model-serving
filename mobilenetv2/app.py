import os
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

app = FastAPI()
current_dir = os.path.dirname(os.path.abspath(__file__))
saved_model_path = os.path.join(current_dir, "mobilenetv2")

model = tf.saved_model.load(saved_model_path)

serving_fn = model.signatures["serving_default"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
        
        if image.mode == "RGBA":
            image = image.convert("RGB")

        image = image.resize((224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0

        input_tensor_name = list(serving_fn.structured_input_signature[1].keys())[0]
        input_data = {input_tensor_name: tf.convert_to_tensor(image)}

        output = serving_fn(**input_data)

        output_tensor_name = list(output.keys())[0]
        prediction = output[output_tensor_name].numpy()
        predicted_class = int(np.argmax(prediction, axis=1)[0])

        return {"result": predicted_class}
    except Exception as e:
        return {"error": str(e)}
