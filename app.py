import os
import io
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image

from model_utils import Network, load_mnist_data, load_model, save_model

MODEL_PATH = "model.pkl"

app = Flask(__name__)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def ensure_model():
    if os.path.exists(MODEL_PATH):
        net = load_model(MODEL_PATH)
    else:
        # Train a small model if not present
        training_data, test_data = load_mnist_data()
        net = Network([784, 64, 64, 10])
        net.sgd(training_data, epochs=3, mini_batch_size=20, eta=3.0, test_data=test_data)
        save_model(net, MODEL_PATH)
    return net


net = ensure_model()
_, test_data = load_mnist_data()
accuracy = net.evaluate(test_data) / len(test_data)


@app.route("/")
def index():
    return render_template("index.html", accuracy=round(accuracy, 4))


def preprocess_image(img: Image.Image) -> np.ndarray:
    # Convert to 28x28 grayscale and invert to match training (white digit on black)
    # Use modern Pillow resampling enum to avoid AttributeError on newer Pillow
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        # Fallback for older Pillow versions
        resample = Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.NEAREST
    img = img.convert("L").resize((28, 28), resample)
    arr = np.asarray(img).astype(np.float32)
    # Normalize and invert: canvas draws black on white; model expects white digit on black
    arr = 255.0 - arr
    arr = arr.reshape(784, 1) / 255.0
    return arr


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("image")
    if not data:
        return jsonify({"error": "no image provided"}), 400
    # data is data-url like 'data:image/png;base64,....'
    _, encoded = data.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes))
    x = preprocess_image(img)
    output = net.feedforward(x)
    probs = (output / output.sum()).flatten()
    pred = int(np.argmax(probs))
    confidences = [float(p) for p in probs]
    return jsonify({"prediction": pred, "confidences": confidences})


if __name__ == "__main__":
    app.run(debug=True)
