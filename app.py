import gradio as gr
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# ----------------------------
# CONFIG
# ----------------------------
IMAGE_SIZE = 224
MODEL_PATH = "mobilenet_final.keras"
CLASSES_PATH = "class_names.json"

# ----------------------------
# LOAD MODEL & CLASSES
# ----------------------------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

print("Loading class names...")
with open(CLASSES_PATH, "r") as f:
    class_names = json.load(f)
print("Classes:", class_names)


# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict(input_image):

    # If Gradio gives numpy array, convert to PIL
    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image)

    # Ensure RGB
    input_image = input_image.convert("RGB")

    # Resize to training size
    input_image = input_image.resize((IMAGE_SIZE, IMAGE_SIZE))

    # Convert to numpy
    img = np.array(input_image, dtype=np.float32)

    # Apply the SAME preprocessing used during training
    from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
    img = preprocess_input(img)

    # Add batch dimension
    img = np.expand_dims(img, 0)

    # Predict
    preds = model.predict(img)[0]

    # Return confidence scores
    return {class_names[i]: float(preds[i]) 
            for i in range(len(class_names))}

    return confidences


# ----------------------------
# GRADIO UI
# ----------------------------
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload Eye Image"),
    outputs=gr.Label(num_top_classes=1, label="Prediction"),
    title="Eye Disease Classification",
    description="Upload a retinal image for disease prediction.",
    allow_flagging="never"
)

# ----------------------------
# RUN (Hugging Face Spaces)
# ----------------------------
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
