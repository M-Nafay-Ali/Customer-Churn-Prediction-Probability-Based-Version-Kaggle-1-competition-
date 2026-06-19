# 🔮 Customer Churn Prediction Dashboard
An end-to-end Machine Learning project that predicts the probability of customer churn using advanced gradient boosting models and deploys the optimal solution via a live interactive Streamlit web dashboard.
## 🚀 Live Demo
Check out the interactive web application live here: **[https://userchurn.streamlit.app/]**
---
## 📌 Project Overview
Customer retention is a vital metric for subscription-based businesses. This project utilizes dataset records from the **Kaggle Playground Series (Season 6, Episode 3)** to construct an automated machine learning workflow capable of accurately distinguishing between loyal customers and those at risk of leaving.
The project encompasses raw data ingestion, robust pipeline preprocessing, evaluation of multiple state-of-the-art algorithms, model ensembling, and interactive cloud deployment.
---
## 📊 Dataset & Features
The dataset contains **594,194 records** with 19 structural feature components monitoring customer accounts:
* **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
* **Account Information:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
* **Services Signed Up For:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
* **Target Variable:** `Churn` (Converted from string values `Yes`/`No` to binary integers `1`/`0`)
---
## 🏗️ Modeling Workflow & Architecture
Instead of using manual, error-prone data mutations, this framework implements an automated Scikit-Learn `ColumnTransformer` pipeline layout:
1.  **Numerical Processing:** Continuous numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`) are scaled via `StandardScaler` to handle variance discrepancies.
2.  **Categorical Processing:** Text columns are dynamically mapped using `OneHotEncoder(handle_unknown='ignore')`.
3.  **Stratified Splitting:** Splitting data into 80/20 train/validation splits while maintaining class ratios (77.5% loyal / 22.5% churned).
### 🏆 Model Comparison (Local Validation Results)
Three gradient boosting architectures were trained and evaluated using the **ROC AUC** metric:

| Model Architecture | Local Validation ROC AUC |
| :--- | :--- |
| **LightGBM Classifier** | **0.91367** 🏆 *(Best Individual Model)* |
| XGBoost Classifier | 0.91363 |
| CatBoost Classifier | 0.91285 |
| Blended Ensemble Average | 0.91361 |

*The optimized LightGBM model scored a **0.91152** on the private Kaggle leaderboard.*
---
## 🖥️ Streamlit Web Application
The final production pipeline was exported into a serialized format (`best_churn_model.pkl`) using `joblib` and integrated into a responsive Streamlit layout.
### App Features:
* **Dynamic Sliders & Inputs:** Adjust user attributes like tenure, payment methods, and pricing constraints interactively.
* **Real-time Inference:** Passes real-time user selections through the pipeline engine instantly.
* **Visual Risk Gauges:** Provides clear color-coded warning feedback classifying retention metrics into *High Risk* or *Low Risk*.
* **Link:-**(https://userchurn.streamlit.app/)


## 📞 Contact:-
* **LinkedIn:-**[https://www.linkedin.com/in/mohammed-nafay-ali-16519138a?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app]

* **Email:-**[englandengland271@gmail.com]
