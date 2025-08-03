import tensorflow as tf
import gradio as gr
import numpy as np

# Load your trained model (ensure this file is present in the same directory)
model = tf.keras.models.load_model("cxr_pneumonia_model.h5")

def predict_xray(img):
    img = img.resize((224,224))
    x = np.array(img).astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)[0][0]
    return "PNEUMONIA" if pred > 0.5 else "NORMAL"

iface = gr.Interface(
    fn=predict_xray,
    inputs=gr.Image(type="pil"),
    outputs=gr.Text(),
    title="Chest X-ray Pneumonia Detection",
    description="Upload a chest X-ray image to predict PNEUMONIA or NORMAL."
)

if __name__ == "__main__":
    iface.launch()
