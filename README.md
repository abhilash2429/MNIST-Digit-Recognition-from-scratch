🧠 ## MNIST Digit Draw & Predict

A small web app where you can draw a digit (0–9) and the app will predict it using a simple neural network trained on the MNIST dataset — built completely from scratch (no TensorFlow/Keras models, just NumPy math and some Python magic ✨).

🚀 What’s Inside

model_utils.py – A tiny neural network implementation (feedforward + backprop + SGD) using only NumPy.

train_and_save.py – Trains the model on MNIST and saves it as model.pkl.

app.py – A Flask app that loads the model and serves predictions through a web interface.

static/app.js & static/style.css – Frontend for drawing digits and displaying confidence levels.

requirements.txt – Python dependencies.

⚙️ Setup & Run

Clone the repo

git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>


Install dependencies

pip install -r requirements.txt


Train the model

python train_and_save.py


This trains a small neural net (784 → 64 → 64 → 10) and saves it as model.pkl.

Run the web app

python app.py


Then open your browser at 👉 http://127.0.0.1:5000

🖌️ Try It Out

Draw a digit on the canvas

Hit Predict

See the model’s guess and confidence levels instantly

If you don’t see predictions, check the console or Flask logs for any errors.
