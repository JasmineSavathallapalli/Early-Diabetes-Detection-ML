Early Detection of Diabetes Using Predictive Analytics and Machine Learning Models

This project aims to **predict the early onset of diabetes** using various **supervised machine learning algorithms**. It uses a structured dataset with real-world health parameters to detect whether a person is likely to be diabetic or not. Multiple models were trained, evaluated, and compared for performance.

---

##  Project Highlights

- 🚀 Built and trained 10+ ML models for binary classification
- 📊 Compared models using accuracy, precision, recall, F1-score, and ROC-AUC
- 📈 Visualized confusion matrices and ROC curves
- 🧪 Validated predictions using real-time new user input

---

##  Features Used

| Feature                   | Description                                     |
|---------------------------|-------------------------------------------------|
| **Glucose**               | Plasma glucose concentration (2 hours in OGTT)  |
| **BloodPressure**         | Diastolic blood pressure (mm Hg)                |
| **SkinThickness**         | Triceps skin fold thickness (mm)                |
| **Insulin**               | 2-Hour serum insulin (mu U/ml)                  |
| **BMI**                   | Body mass index (weight in kg/(height in m)^2)  |
| **DiabetesPedigreeFunction** | Diabetes likelihood based on family history  |
| **Age**                   | Age in years                                    |
| **Pregnancies**           | Number of times pregnant                        |
---

##  Machine Learning Models Used

| **Model**                        | **Accuracy** |
| -------------------------------- | ------------ |
| **Logistic Regression**          | 75.32%       |
| **Random Forest**                | 73.37%       |
| **Support Vector Machine (SVM)** | 74.67%       |
| **XGBoost Classifier**           | 71.42%       |
| **Gradient Boosting**            | 75.97%       |
| **Extra Trees Classifier**       | 76.62%       |
| **LightGBM Classifier**          | 72.00%       |
| **AdaBoost Classifier**          | 75.32%       |
| **K-Nearest Neighbors (k=12)**   | 79.22%       |
| **MLP Neural Network**           | 76.62%       |
| **Gaussian Naive Bayes**         | 75.32%       |
| **Decision Tree Classifier**     | 71.40%       |

---

## 🛠 Libraries Used

- `pandas`, `numpy` — Data manipulation
- `scikit-learn` — ML algorithms, preprocessing, evaluation
- `matplotlib`, `seaborn` — Visualization
- `xgboost`, `lightgbm` — Advanced ensemble models
- `joblib` — Model saving & loading

## Evaluation
Each model is evaluated using:
- Accuracy
- Confusion Matrix
- Classification Report
- ROC Curve
- New input prediction

##  Files Included
- `diabetes_prediction.ipynb`: Main notebook
- `diabetes.csv : dataset
- `.pkl` files: Saved trained models
- `README.md`: Project overview

## 🧪 Predict on New Data

You can input your own values for features like `Glucose`, `BMI`, `Age`, etc., and the models will return whether the person is **Diabetic** or **Not Diabetic**.

---

##  Demo
Test real inputs and get prediction using the trained models.

##  Conclusion
Among 14 tested models, K-Nearest Neighbors (k=12) achieved the highest accuracy (79.22%), followed by MLP Neural Network and Extra Trees Classifier (76.62%). Supervised models outperformed unsupervised ones, confirming their suitability for predictive healthcare tasks like diabetes diagnosis.

---

## 👩‍💻 Author

**Jasmine Savathallapalli**

## ⭐ Star this repository if you found it useful!
