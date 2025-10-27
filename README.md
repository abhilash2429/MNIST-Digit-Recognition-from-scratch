# (0-9) MNIST Digit Draw & Predict

I know this Project is much Older than me , But as a newbie just getting into Deep Learning , Learning Perceptrons and Neural Netwoks.
This project was fun as im testing my computer to recognize digits like i used to test small kids when they start to learn digits and Alphabets😂


I have Implemented SGD on a tiny neural network with two hidden layers. I have implemented it from scratch using only NumPY , no tensorflow and keras 👍.
The Network had achieved a terrible 95.4% Accuracy and sometimes may recognize 7 as 3 or 4 whatever 
Looking Forward to learn CNN's and wil try to achieve a 98+ Accuracy.


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

 

 
 
