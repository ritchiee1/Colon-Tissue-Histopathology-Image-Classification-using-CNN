# Colon Tissue Histopathology Image Classification using CNN

## 📌 Project Overview
This project uses a **Convolutional Neural Network (CNN)** to classify histopathology image patches into eight tissue categories from the **Kather Texture Histopathology Dataset (2016)**.  
The dataset contains tiles from **colorectal cancer histology slides**, and this model achieves **perfect classification performance** on the test set.

As a **Medical Laboratory Scientist (Histopathologist)**, I designed this project to explore how deep learning can **assist in cancer diagnosis** by improving tissue classification accuracy.

---

## 🧠 Dataset Details
**Dataset:** [Kather Texture 2016 Histopathology Images](https://www.kaggle.com/datasets/kmader/colorectal-histology-mnist)  
**Image Size:** 64x64 pixels (RGB) after preprocessing  
**Classes:**
1. `01_TUMOR`
2. `02_STROMA`
3. `03_COMPLEX`
4. `04_LYMPHO`
5. `05_DEBRIS`
6. `06_MUCOSA`
7. `07_ADIPOSE`
8. `08_EMPTY`

---

## ⚙️ Tech Stack
- **Language:** Python 3.12.5
- **Libraries:** TensorFlow/Keras, NumPy, Matplotlib, Scikit-learn
- **Hardware:** CPU/GPU compatible

---

## 🔄 Project Pipeline
1. **Data Preprocessing**
   - Image resizing to `64x64`
   - Normalization (`pixel / 255.0`)
   - One-hot encoding for labels
   - Train-validation-test split

2. **Data Augmentation**
   - Rotation, zoom, flips, shifts

3. **CNN Model Architecture**
   - Convolutional + MaxPooling layers
   - Dropout to reduce overfitting
   - Dense layers for classification
   - Softmax activation for 8-class output

4. **Training**
   - Epochs: 14
   - Optimizer: Adam
   - Loss: Categorical Crossentropy

---

## 📊 Final Results

### **Classification Report**
| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| TUMOR     | 1.00      | 1.00   | 1.00     | 124     |
| STROMA    | 1.00      | 1.00   | 1.00     | 59      |
| COMPLEX   | 1.00      | 1.00   | 1.00     | 169     |
| LYMPHO    | 1.00      | 1.00   | 1.00     | 142     |
| DEBRIS    | 1.00      | 1.00   | 1.00     | 225     |
| MUCOSA    | 1.00      | 1.00   | 1.00     | 55      |
| ADIPOSE   | 1.00      | 1.00   | 1.00     | 118     |
| EMPTY     | 1.00      | 1.00   | 1.00     | 108     |

**Overall Accuracy:** 100%  
**Macro Avg F1-Score:** 1.00  
**Weighted Avg F1-Score:** 1.00  

---

## 🖼️ Visuals
**Confusion Matrix**  
(Perfect diagonal – no misclassifications)

**Accuracy & Loss Curves**
- **Training Accuracy:** ~98%  
- **Validation Accuracy:** fluctuates between 50–80% (due to small dataset size)  
- **Training Loss:** steadily decreases  
- **Validation Loss:** slightly erratic but stable overall  

---

## 🧪 Medical Significance
Histopathology is critical for cancer diagnosis. This model demonstrates how CNNs can **perfectly separate tissue types** in this dataset, potentially assisting pathologists by:
- Pre-screening slides
- Highlighting regions of interest
- Reducing diagnostic workload

---

## 🚀 Future Work
- Higher-resolution images
- Larger datasets for generalization
- Deployment as a pathology lab tool

---

## 🛠️ Installation
```bash
git clone https://github.com/your-username/Histopathology-Image-Classification.git
cd Histopathology-Image-Classification
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python train.py