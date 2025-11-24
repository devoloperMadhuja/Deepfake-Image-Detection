<div align="center">

# 🧠 Deepfake Image Detection System  
### CNN • GAN • Diffusion Models • Flask Web App  

Detecting AI-generated (fake) images and identifying *which generative model created them*.

---

![AI Banner](https://img.shields.io/badge/Deepfake%20Detection-AI%20Forensics-blueviolet?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

# 📌 **Project Overview**

Deepfakes have become a major cybersecurity threat.  
This project uses **Convolutional Neural Networks (CNNs)**, **GAN datasets**, and **Diffusion Model datasets** to detect whether an image is *real or AI-generated* and attributes the fake image to one of the generative models:

- **StyleGAN**  
- **MinDALL·E**  
- **OpenJourney**  
- **Stable Diffusion**

The system includes:

✔ A **deep-learning model** for real vs. fake  
✔ A **multi-class attribution model**  
✔ A **Flask web application** for real-time detection  
✔ A dataset of **170,000+ images**

---

## 📂 Repository Structure

Deepfake-Image-Detection/
│── dataset/ # Real + Fake images (GAN & Diffusion)
│── models/ # Trained PyTorch models
│── notebooks/ # Colab notebooks used for training
│── src/ # CNN architectures, utilities
│── flask_app/
│ ├── static/
│ ├── templates/
│ └── app.py
│── requirements.txt
│── README.md

---

# 🎯 **Features**

### 🧠 Deep Learning  
- Trained CNN models: **SimpleCNN**, **ResNet18**, **ResNeXt50**  
- Binary classification: **Real vs Fake**  
- Multi-class generator attribution  
- Over **170K training images** (GAN + Diffusion)

### 🌐 Web Application  
- Flask-based UI  
- Upload any image → Get prediction instantly  
- Shows **confidence score** and **model attribution**

### 📊 Evaluation  
- Precision, Recall, F1-score  
- Confusion Matrix  
- Accuracy: **95%+ on validation**

---

# 🧰 **Tech Stack**

**Languages:** Python  
**Frameworks:** PyTorch, Flask  
**DL Models:** CNN, ResNet18, ResNeXt50  
**Tools:** Google Colab, NumPy, Pandas, OpenCV  
**Datasets:** GAN-based & Diffusion (Kaggle)

---

## 📂 Datasets Used

This project uses two Kaggle datasets:

### 🔹 1. Real vs Fake Images (GAN-based)
Kaggle Dataset Link: [https://www.kaggle.com/...  ](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
Contains real and GAN-generated images including StyleGAN and other models.

### 🔹 2. Diffusion Model Generated Images
Kaggle Dataset Link: [https://www.kaggle.com/... ](https://www.kaggle.com/datasets/jacobheldt/syntheticeye-diffusion-faces) 
Contains images generated using Stable Diffusion, OpenJourney, and MinDALL·E.

Both datasets combined give:
- **170,000+ total images**
- Real + style-GAN + Diffusion images(min-dalle, openjourney, stable-diffusion)
- Balanced distribution for multi-class attribution



---

# 🗂️ **Dataset Details**

The dataset consists of:

### ✔ **Real images**
Scraped / collected from open datasets

### ✔ **GAN-generated**
- StyleGAN  
- MinDALL·E  

### ✔ **Diffusion-generated**
- Stable Diffusion  
- OpenJourney  

Dataset Distribution:
Total Images: 170,000+
Real Images: ~70,000
Fake Images (GAN): ~60,000
Fake Images (Diffusion): ~40,000

# 🏗️ **System Architecture**

            ┌──────────────────────────┐
            │      Input Image          │
            └─────────────┬────────────┘
                          │
                          ▼
              ┌────────────────────┐
              │  Preprocessing     │
              │ (resize, normalize)│
              └──────────┬─────────┘
                         ▼
           ┌──────────────────────────┐
           │    CNN Classification     │
           │  (SimpleCNN / ResNet18)   │
           └─────────────┬────────────┘
                         │
                         ▼
          ┌──────────────────────────┐
          │  GAN/Diffusion Attribution│
          └─────────────┬────────────┘
                         ▼
           ┌──────────────────────────┐
           │  Flask Web Application    │
           └──────────────────────────┘


---

# 🚀 How to Run Locally

### 1️⃣ Clone the repo
```bash
git clone https://github.com/<your-username>/Deepfake-Image-Detection.git
cd Deepfake-Image-Detection
```
### 2️⃣ Install Dependencies
Make sure you have Python 3.8+ installed.
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Flask Web Application
```bash
python flask_app/app.py
```

### 4️⃣ Open browser
```bash
http://127.0.0.1:5000
```
---

# 🧪 Model Training (Google Colab)

- Runtime: GPU (Tesla T4 / V100)
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Augmentations:
      **Horizontal Flip**
      **Rotation**
      **Color Jitter**
      **Random Erase**

---
# 📊 Results

| Model     | Accuracy |
| --------- | -------- |
| SimpleCNN | 99.2%    |
| ResNet18  | 80.12%   |
| ResNeXt50 | 83.72%   |

---
## 📸 Screenshots

### 📌 Screenshot 1 — Web App Interface  
![Web App](https://github.com/devoloperMadhuja/Deepfake-Image-Detection/blob/main/Web%20App%20Interface%20.jpg)

### 📌 Screenshot 2 — StyleGAN Fake Image Prediction  
![StyleGAN Prediction](https://github.com/devoloperMadhuja/Deepfake-Image-Detection/blob/main/StyleGAN%20prediction%20screenshot.jpg)

### 📌 Screenshot 3 — MinDALL·E Fake Image Prediction  
![MinDALLE Prediction](https://github.com/devoloperMadhuja/Deepfake-Image-Detection/blob/main/MinDALL%C2%B7E%20prediction%20screenshot.jpg)

### 📌 Screenshot 4 — OpenJourney Fake Image Prediction  
![OpenJourney Prediction](https://github.com/devoloperMadhuja/Deepfake-Image-Detection/blob/main/OpenJourney%20prediction%20screenshot.jpg)

### 📌 Screenshot 5 — Stable Diffusion Fake Image Prediction  
![Stable Diffusion Prediction](https://github.com/devoloperMadhuja/Deepfake-Image-Detection/blob/main/Stable%20Diffusion%20prediction%20screenshot.jpg)

---

# 🌱 Future Enhancements

- Video deepfake detection
- API endpoint for enterprise integration
- Mobile version (Flutter)
- Lightweight model deployment
- Real-time face manipulation detection

---
# 👩‍💻 Developer
- Madhuja Deb Adhikari
- B.Tech — CSE (Cyber Security)
- Rashtriya Raksha University
- GitHub: https://github.com/devoloperMadhuja

---
# 📜 License
This project is licensed under the MIT License.
