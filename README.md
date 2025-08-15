# Crime Severity Prediction using Machine Learning & Deep Learning

## Project Overview
This project predicts the severity level of crimes (Low, Medium, High) using crime dataset of India which we found on Kaggle.  
The aim is to assist in better resource allocation and strategic decision-making for law enforcement agencies.

We experimented with **a variety of supervised and deep learning models** — including hyperparameter tuning (Optuna),  
sampling techniques, threshold tuning, and ensemble methods, to achieve optimal results.

---

## Repository Structure
- **eandpfinal.py** → EDA & preprocessing steps.
- **deeplearning.py** → Deep learning model architectures & training.
- **supervised_models.py** → Supervised learning models with various tuning strategies.

---

## Technologies Used
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, TensorFlow/Keras, XGBoost, LightGBM, CatBoost, Matplotlib, Seaborn, Imbalanced-learn  
- **Tuning:** Optuna  
- **Techniques:** Class weighting, threshold tuning, hybrid sampling

---

## Dataset
- Source: *https://drive.google.com/file/d/1KryiT7NScbLWeGjuOakjgMRiGMuf3A74/view?usp=sharing*  
- Target variable: Crime Severity Level (0 = High, 1 = Low, 2 = Medium)  
- Features: Crime time, weapon type, location details, police deployment, etc.  
- Class imbalance handled via SMOTE, Tomek Links, focal loss, and threshold tuning.

---

## Results
| Model                                        | Accuracy | Macro F1 | F1 (Class 0) | F1 (Class 1) | F1 (Class 2) |
| -------------------------------------------- | -------- | -------- | ------------ | ------------ | ------------ |
| **Direct CatBoost (Tuned)**                  | **0.7930** | 0.7100   | **0.8800**   | 0.6200       | 0.6400       |
| OvR CatBoost (Tuned Base)                    | 0.7902   | 0.7203   | 0.8729       | 0.6424       | **0.6457**   |
| OvR XGBoost (Default Base)                   | 0.7864   | 0.7020   | 0.8752       | 0.6265       | 0.6042       |
| Direct XGBoost (Tuned)                       | 0.7883   | **0.7210** | 0.8726       | **0.6500**   | 0.6405       |
| OvR LightGBM (Default Base)                   | 0.7923   | 0.7198   | 0.8756       | 0.6433       | 0.6403       |
| Stacking Classifier LR Final (Untuned)        | 0.7911   | 0.7037   | 0.8796       | 0.6112       | 0.6203       |
| Stacking Classifier LR Final (Tuned)          | 0.7912   | 0.7039   | 0.8797       | 0.6112       | 0.6209       |
| Decision Tree (Default)                       | 0.7737   | 0.6699   | 0.8716       | 0.5891       | 0.5489       |
| Decision Tree (Tuned)                         | 0.7819   | 0.6947   | 0.8735       | 0.6151       | 0.5954       |

---

### Deep Learning Experiments

| Model                                        | Approach & Changes                                      | Accuracy | Macro F1 | F1 (High) | F1 (Low) | F1 (Medium) |
| -------------------------------------------- | ------------------------------------------------------- | -------- | -------- | --------- | -------- | ----------- |
| **Keras Baseline**                           | Hybrid Sampling (SMOTE + Tomek) + Focal Loss            | **0.79** | **0.71** | **0.88**  | **0.65** | 0.60        |
| Keras Baseline                               | Weighted Focal Loss (Improve Medium Recall)             | 0.76     | 0.68     | 0.85      | 0.57     | 0.63        |
| Keras Baseline                               | Weighted Focal Loss + Threshold Tuning (Improve Medium) | 0.76     | 0.69     | 0.85      | 0.59     | **0.63**    |
| Keras Baseline                               | Increased Low Class Weight (Improve Low)                | 0.78     | 0.65     | **0.88**  | 0.60     | 0.48        |

---

## Installation & Usage
```bash
# Clone the repository
git clone https://github.com/Shaivi1706/crime-severity-analysis.git
cd crime-severity-analysis

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebooks or scripts
jupyter notebook
