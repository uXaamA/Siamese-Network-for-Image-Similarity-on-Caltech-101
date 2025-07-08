# 🧠 Siamese Network for Image Similarity on Caltech-101 Dataset

This project implements a **Siamese Neural Network** from scratch using **PyTorch**, built for **learning image similarity** between pairs of images. The model was trained and evaluated on the **Caltech-101** dataset and uses a **contrastive loss function** to learn an embedding space where similar images are close together, and dissimilar ones are far apart.

---

## 🎯 Objective

The primary goal of this project is to train a **deep learning model that can tell whether two images belong to the same class or not** — without relying on classification directly. This approach is widely used in:
- Face verification (e.g., Face ID)
- Signature verification
- Product similarity in e-commerce
- Image retrieval

---

## 📂 Project Structure
project/
- ├── rollNumber_02_task1.py # End-to-end integration script (training, testing, visualization)
- ├── train.py # Training loop and checkpointing
- ├── test.py # Evaluation script
- ├── model.py # Siamese network class using ResNet-50
- ├── data_utils.py # Dataset class and custom dataloader for image pairs
- ├── weights/ # Directory to store trained model weights
- ├── graphs/ # Visualizations (loss curves, confusion matrix, etc.)
- ├── Report.pdf # Final report with training analysis
- ├── requirements.txt # Project dependencies
- └── README.md # You are here 📘



---

## 🗃 Dataset: Caltech-101

- Contains **9,145 images** across **102 classes**
- Classes include objects like helicopters, faces, animals, instruments, etc.
- Dataset is stored in a **single directory**, so we wrote a **custom split function** to separate into:
  - `Train` set
  - `Validation` set
  - `Test` set  
- Splitting preserves **class ratios** to maintain balance.

📎 **Reference**: [Caltech-101 Dataset Page](https://data.caltech.edu/records/mzrjq-6wc02)

---

## 🔀 Data Preparation & Preprocessing

- Each training sample is a **pair of images**
- Label is `0` if the two images are similar (same class), and `1` if dissimilar (different classes)
- To avoid imbalance, we sample an equal number of positive and negative pairs
- **Transforms** applied:
```python
transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
]) ```
```


## 📐 What the Model Learns

Rather than learning “what is a tiger?”, the model learns:
- “How similar are two things in their deep visual patterns?”

That’s powerful because:
- It doesn’t rely on labeled classes during inference
- It generalizes to *any* pair — even unseen categories

---

## 📌 Contrastive Loss — Mathematical Form

We use **contrastive loss** to train the network:

\[
L = \frac{1}{2N} \sum_{i=1}^N \left[ y_i \cdot d_i^2 + (1 - y_i) \cdot \max(\alpha - d_i, 0)^2 \right]
\]

Where:

- \( y_i \in \{0, 1\} \):  
   - 0 → same class  
   - 1 → different class  
- \( d_i = \| e_{a_i} - e_{p_i} \|_2 \): Euclidean distance between embeddings
- \( \alpha \): margin (usually set to 1.0)

This loss:
- Reduces distance between similar image pairs
- Increases distance for dissimilar pairs (up to margin)

## Graphs
![accuracy_plot](https://github.com/user-attachments/assets/491f8188-3f18-4f4b-86fb-5a5025d06abc)
![recall_plot](https://github.com/user-attachments/assets/5c933d42-3ba7-4cca-b3fd-90488f07e6d8)
![f1_score_plot](https://github.com/user-attachments/assets/9f458247-7109-4ab7-9862-439d37a72149)
![loss_plot](https://github.com/user-attachments/assets/74cb3e38-ca32-4a89-8093-6818c7dee00a)
![confusion_matrix_train](https://github.com/user-attachments/assets/05dba442-6204-4ef6-bed4-6acf79b16d64)
![confusion_matrix_val](https://github.com/user-attachments/assets/ee608a3b-3938-420e-8b12-8c8aa61c0b35)



---

👤 Author
Muhammad Usama
🎓 MS Data Science — Information Technology Lahore
⚙️ Technical Data Analyst, Haier Pakistan
🧠 Passion: AI | ML | Computer Vision | Startups
🔥 Founder: Phantomwears Clothing
📧 usaman3244015@gmail.com
📍 Lahore, Pakistan
