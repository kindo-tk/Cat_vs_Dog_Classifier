# Cat vs Dog Classifier
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Deep_Learning-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-ff4b4b)
![NumPy](https://img.shields.io/badge/NumPy-Array_Processing-lightblue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green)

---

## Overview

This project implements a **Cat vs Dog Image Classifier** using **Deep Learning**. 

It combines:
- A **Custom CNN with Regularization**, and  
- Two **Pretrained Transfer Learning Models** — **MobileNetV2** and **EfficientNetB0**,  
trained and fine-tuned using the **two-phase fine-tuning strategy**.

The models are compared based on accuracy and AUC scores, and the **best-performing model (EfficientNetB0)** is deployed through a **Streamlit web app**, where users can upload an image and get instant predictions.

---

## Problem Statement

The goal is to **classify an image as either a Cat or a Dog** using convolutional neural networks.  
This problem, though simple, tests the robustness of deep learning models in real-world object recognition tasks where:

- Training data can vary in **lighting, pose, and background**.  
- The model must generalize well to **unseen samples**.  
- Transfer learning plays a vital role in **reducing training time** while maintaining high accuracy.

---
##  Project Pipeline

### **1️⃣ Data Handling & Preprocessing**
- Dataset sourced from the <a href = "https://www.kaggle.com/datasets/tufankundu/dog-vs-cat" target='blank'>**Kaggle Dogs vs Cats** dataset.
- Images organized into `train/val/test` splits (70/10/20).
- Data augmentation: rotation, shifts, zoom, flips, and normalization.

### **2️⃣ Model Architectures**
- **Custom CNN:** Designed from scratch with L2 regularization, dropout, and batch normalization.  
- **MobileNetV2:** Lightweight pretrained model fine-tuned in two phases.  
- **EfficientNetB0:** State-of-the-art pretrained model, fine-tuned with discriminative learning rates.

### **3️⃣ Training Strategy**
- Phase 1: Train only the classification head (frozen base).
- Phase 2: Fine-tune the last 20 layers of the pretrained base.
- Metrics: **Accuracy** and **AUC (Area Under ROC Curve)**.
- Early stopping and learning rate scheduling are used for optimal convergence.

### **4️⃣ Model Comparison**
| **Model** | **Accuracy** | **AUC Score** |
|------------|---------------|---------------|
| EfficientNetB0 | 0.9916   | 0.9997  |
| MobileNetV2    | 0.9874   | 0.9994  |
| Custom CNN     | 0.9692   | 0.9961  |

---

##  Final Model: EfficientNetB0

- Selected based on **highest validation AUC and accuracy**.
- Saved as:
  - `best_overall.keras` (full model)
  - `best_overall.weights.h5` (weights)
- Deployed in **Streamlit** app for real-time predictions.

---
## Streamlit Web App

### 🔗 **Live App:** [Click Here to Try the Classifier](https://catvsdogclassifierbytk.streamlit.app/)  

###  Features
- Let the user upload any `.jpg` / `.png` image of a Cat or Dog.
- Get **real-time predictions** with confidence score.
- Runs directly in the browser using **Streamlit**.

###  How It Works
1. Loads the trained **EfficientNetB0** model (`best_overall.keras`).
2. Preprocesses the image using `EfficientNetB0`’s preprocessing.
3. Outputs a prediction with confidence.

---

## Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/kindo-tk/Cat_vs_Dog_Classifier.git
```

2. Navigate to the project directory:

```bash
cd cat-vs-dog-classifier
```

3. Create a virtual environment:

```bash
python -m venv .venv
```

4. Activate the virtual environment:

```bash
.venv\Scripts\activate
```

5. Install the required packages:

```bash
pip install -r requirements.txt
```

6. Run the Streamlit application:

```bash
streamlit run app.py
```

7. Open your browser and go to:

```
http://localhost:8501
```

---

##  Technologies Used

| Category | Tools / Frameworks |
|-----------|--------------------|
| Programming | Python |
| Deep Learning | TensorFlow, Keras |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit |
| Utilities | PIL, Scikit-learn |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

For any inquiries or feedback, please contact:

- [Tufan Kundu (LinkedIn)](https://www.linkedin.com/in/tufan-kundu-577945221/)  
- Email: tufan.kundu11@gmail.com  

---

### Screenshots

<img src="https://github.com/kindo-tk/images/blob/main/cat_vs_dog_classifier/Screenshot%20(52).png" width="600">
<img src="https://github.com/kindo-tk/images/blob/main/cat_vs_dog_classifier/Screenshot%20(53).png" width="600">
<img src="https://github.com/kindo-tk/images/blob/main/cat_vs_dog_classifier/Screenshot%20(54).png" width="600">
<img src="https://github.com/kindo-tk/images/blob/main/cat_vs_dog_classifier/Screenshot%20(55).png" width="600">

