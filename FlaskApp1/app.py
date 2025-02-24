from flask import Flask, request, render_template
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load your ONNX model (two-logit output: [1, 2])
session = ort.InferenceSession('beit_finetuned_model.onnx')

def preprocess_image(image):
    # Resize to 224x224 => adjust if your model is different
    image = image.resize((224, 224))
    # Convert to float32 & normalize to [0..1]
    arr = np.array(image).astype(np.float32) / 255.0
    # Transpose to [C, H, W]
    arr = np.transpose(arr, (2, 0, 1))
    # Add batch dimension => [1, 3, 224, 224]
    arr = np.expand_dims(arr, axis=0)
    return arr

def run_inference(image_array):
    """
    Since your model outputs 2 logits for [non-cancerous, cancerous],
    we do softmax => pick highest prob => return (label, cancer_prob).
    """
    input_name = session.get_inputs()[0].name
    # Model returns a single output with shape [1, 2]
    logits = session.run(None, {input_name: image_array})[0]  # e.g. [[-2.10, 1.24]]

    # Softmax
    exps = np.exp(logits)             # shape [1, 2]
    sums = np.sum(exps, axis=1, keepdims=True)
    probs = exps / sums               # shape [1, 2]

    # The second index (probs[0][1]) is the probability of cancer
    cancer_prob = probs[0][1]
    # Class 0 => non-cancerous, Class 1 => cancerous
    pred_class = np.argmax(probs, axis=1)[0]

    # Decide label
    if pred_class == 1:
        label = 'Positive'  # cancerous
    else:
        label = 'Negative'  # non-cancerous

    return label, cancer_prob

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Open image via PIL
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
            # Preprocess
            processed_image = preprocess_image(image)
            # Run inference => get (label, probability)
            label, cancer_prob = run_inference(processed_image)

            # Pass them to result.html
            return render_template(
                'result.html',
                result=label,
                prob=cancer_prob,
                brand_name="SkinGuardian"
            )
    # GET => show index.html
    return render_template('index.html', brand_name="SkinGuardian")

@app.route('/about')
def about():
    return render_template('about.html', brand_name="SkinGuardian")

@app.route('/contact')
def contact():
    return render_template('contact.html', brand_name="SkinGuardian")

if __name__ == '__main__':
    app.run(debug=True)
