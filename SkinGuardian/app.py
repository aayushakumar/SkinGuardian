import io
import numpy as np
from PIL import Image
from flask import Flask, request, render_template
import onnxruntime as ort
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
template_folder = os.path.join(BASE_DIR, 'templates')
app = Flask(__name__, template_folder=template_folder)

if getattr(sys, 'frozen', False):
    # Running in a PyInstaller EXE
    BASE_PATH = sys._MEIPASS
else:
    # Running in normal Python
    BASE_PATH = os.path.dirname(__file__)

model_path = os.path.join(BASE_PATH, "beit_finetuned_model.onnx")
session = ort.InferenceSession(model_path)

# 1) Load ONNX Model
# session = ort.InferenceSession("beit_finetuned_model.onnx")

# 2) Preprocess (adjust to your model's exact needs)
def preprocess_image(pil_image):
    image = pil_image.convert("RGB").resize((224, 224))
    arr = np.array(image).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # [C,H,W]
    arr = np.expand_dims(arr, axis=0)   # [1,C,H,W]
    return arr

# 3) Inference logic: 2-logit model => [non-cancerous, cancerous]
def run_inference(image_array):
    # shape => [1, 3, 224, 224]
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: image_array})[0]  # shape: [1, 2]

    # Softmax
    exps = np.exp(logits)
    sums = np.sum(exps, axis=1, keepdims=True)
    probs = exps / sums  # shape [1,2]

    cancer_prob = probs[0][1]     # Probability of cancer
    pred_class = np.argmax(probs) # 0 or 1

    if pred_class == 1:
        label = "Positive"
    else:
        label = "Negative"
    return label, cancer_prob

# 4) Main Route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            # Load & preprocess
            pil_image = Image.open(io.BytesIO(file.read()))
            arr = preprocess_image(pil_image)
            label, cancer_prob = run_inference(arr)

            return render_template(
                "result.html",
                result=label,
                prob=cancer_prob,
                brand_name="SkinGuardian"
            )
    # GET => show main page
    return render_template("index.html", brand_name="SkinGuardian")

# Additional routes
@app.route("/about")
def about():
    return render_template("about.html", brand_name="SkinGuardian")

@app.route("/contact")
def contact():
    return render_template("contact.html", brand_name="SkinGuardian")


def start_server():
    """Helper function if needed for PyInstaller."""
    app.run(debug=False, host="0.0.0.0", port=5000)

if __name__ == "__main__":
    # Normal local development: python app.py
    app.run(debug=True)
