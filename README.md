# (0-9) MNIST Digit Draw & Predict

I know this project is much older than me, but as a newbie just getting into Deep Learning, learning about Perceptrons and Neural Networks, this was a fun experiment. It felt like testing my computer to recognize digits — just like how we test little kids when they’re learning numbers and alphabets 😂

I implemented Stochastic Gradient Descent (SGD) on a tiny neural network with two hidden layers, built completely from scratch using NumPy (no TensorFlow or Keras 👍). The network achieved a “terrible” 95.4% accuracy, and sometimes mistakes 7 for 3 or 4 — whatever 😅.

Looking forward to diving into CNNs next and hopefully pushing that accuracy past 98%!

---

## What’s Inside

| File | Description |
|------|--------------|
| `model_utils.py` | A tiny neural network implementation (feedforward + backprop + SGD) using only NumPy. |
| `train_and_save.py` | Trains the model on MNIST and saves it as `model.pkl`. |
| `app.py` | A Flask app that loads the model and serves predictions through a web interface. |
| `static/app.js` & `static/style.css` | Frontend for drawing digits and displaying confidence levels. |
| `requirements.txt` | Python dependencies. |

---

## ⚙️ Setup & Run

### 1️⃣ Clone the repo
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train the model
```bash
python train_and_save.py
```
This trains a small neural net (784 → 64 → 64 → 10) and saves it as `model.pkl`.

### 4️⃣ Run the web app
```bash
python app.py
```

Then open your browser and visit 👉 [http://127.0.0.1:5000] 

---

## 🖌️ Try It Out

1. Draw a digit on the canvas  
2. Hit **Predict**  
3. See the model’s **guess and confidence levels** instantly 🎯  

If predictions don’t show up, check your **Flask logs** or browser console for details.

---

## 💡 Note

- This project is designed for learning purposes.  
- The neural net here is intentionally simple — you can improve it by increasing epochs or using convolutional layers.  
- For better accuracy, try implementing a CNN with Keras or PyTorch later.

---

 

 
 
