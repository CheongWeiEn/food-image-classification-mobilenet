# Food Image Classification with MobileNet (Transfer Learning)

This project builds a **food image classifier** using **MobileNet** and transfer learning.  
The goal is to classify food images into multiple categories using a relatively lightweight but powerful convolutional neural network.

---

## 🎯 Objective

Given an input food image, predict the correct **food category** (e.g. noodles, burgers, sushi, steak, etc.) using deep learning.

---

## 📊 Dataset

- ~10,000 labelled food images (multi-class classification)
- Images split into **train**, **validation**, and **test** sets
- Loaded using Keras `ImageDataGenerator` / data loaders  
- Includes real-world variation: lighting, angle, background, etc.

*(If the dataset is from Kaggle or another source, you can add the link here.)*

---

## 🧠 Model – MobileNet Transfer Learning

### Base Model

- Pretrained **MobileNet** (ImageNet weights)
- Lower layers **frozen** initially to keep pretrained features
- Top layers replaced with a custom classification head

### Custom Head

- Global Average Pooling / Flatten layer  
- Dense layers with **ReLU** activation  
- **Dropout** for regularisation (e.g. 0.3)  
- Final Dense layer with **softmax** for multi-class output  

### Training Strategy

1. **Phase 1 – Frozen Base**
   - Train only the new top layers  
   - Faster training, stable start  

2. **Phase 2 – Fine-Tuning**
   - Unfreeze top portion of MobileNet (e.g. last ~40 layers)  
   - Lower learning rate  
   - Train end-to-end to improve accuracy  

### Data Augmentation

- Random rotation  
- Width/height shifts  
- Zoom  
- Horizontal flips  
- Rescaling pixel values  

This helps improve generalisation and reduce overfitting.

---

## 📈 Performance

- Final test accuracy: **~85.6%** on unseen images  
- Correctly classified **8 out of 10** manually tested real-world food photos  
- Training and validation curves show improved performance after fine-tuning MobileNet

*(You can add a confusion matrix screenshot to the repo later in an `images/` folder if you like.)*

---

## 📂 Repository Structure

```text
food_mobilenet_classifier.ipynb
README.md
```
Notebook contains data loading, augmentation, model definition, training, and evaluation.

Dataset is not included in this repository to keep it lightweight; see notebook for loading instructions.

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- MobileNet (transfer learning)
- NumPy, Pandas
- Matplotlib / Seaborn

## 💡 Key Learnings
- Applying transfer learning with a pretrained CNN (MobileNet)
- Using data augmentation and regularisation to reduce overfitting
- Fine-tuning pretrained layers for better performance
- Evaluating model performance on both test data and real-world images

## 🚀 Future Improvements
- Experiment with other pretrained models (EfficientNet, ResNet, DenseNet, Inception)
- Add more food categories and increase dataset diversity
- Deploy the model as a simple web app (e.g. Streamlit) or API
- Use Grad-CAM or similar visualisation tools to see which parts of the image the model focuses on

## 👤 Author
Cheong Wei En
Data Science Student @ Ngee Ann Polytechnic
LinkedIn: https://www.linkedin.com/in/cheong-wei-en-222911303
