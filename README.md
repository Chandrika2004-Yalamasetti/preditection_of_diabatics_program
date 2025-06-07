# Prediabetes Detection Using AI

This project uses machine learning models (XGBoost + KNN) to detect **prediabetes** using **non-laboratory health data**.

## 💡 Project Objective
To build a smart and accessible prediabetes detection system that does not rely on blood reports — suitable for early lifestyle-based screening.

## 🔍 Features
- XGBoost + KNN ensemble model
- Achieved high recall and F1-score
- Built a Flask-based web app for predictions
- Accuracy: **89.43%**, Recall: **93.94%**

## 🧠 Machine Learning Stack
- Python
- XGBoost
- KNN
- scikit-learn
- pandas, numpy, matplotlib

## 🌐 Web Technology
- Flask (Backend)
- HTML/CSS (Frontend)

## 📊 Dataset
- `diabetes.csv` – used for training and evaluation
- Features: Age, BMI, Physical Activity, etc. (non-lab data)

## 📈 Model Performance

| Metric     | Value     |
|------------|-----------|
| Accuracy   | 89.43%    |
| Precision  | 87.32%    |
| Recall     | 93.94%    |
| F1 Score   | 90.51%    |

## 📁 How to Run Locally

```bash
git clone https://github.com/Chandrika2004-Yalamasetti/preditection_of_diabatics_program.git
cd preditection_of_diabatics_program
pip install -r requirements.txt
python app.py
