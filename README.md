# HOT-ROLLING-DEFECT-PREDICTION-CHALLENGE
Machine Learning solution for steel defect prediction using XGBoost. Includes feature engineering, class imbalance handling, model optimization, and production-ready prediction pipeline. Developed by Kaushal Raj.

# 🔍 Tata Steel Defect Prediction Challenge

> Machine Learning solution for predicting manufacturing defects in steel coils using XGBoost, feature selection, threshold optimization, and imbalanced data handling.

**Developed by:** Kaushal Raj

![Python](https://img.shields.io/badge/Python-3.x-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Leaderboard](https://img.shields.io/badge/Best%20Score-84.63-green)

---

# 📌 Problem Statement

The objective of this project is to predict whether a steel coil is defective or non-defective using manufacturing process parameters collected during production.

The challenge involves:

- Highly imbalanced data
- Missing values
- Complex industrial process variables
- Rare defect occurrences

---

# 📊 Dataset Overview

| Metric | Value |
|----------|----------|
| Training Samples | 1352 |
| Test Samples | 339 |
| Features | 49 |
| Target Variable | Y |
| Defect Samples | 66 |
| Non-Defect Samples | 1286 |
| Defect Ratio | 4.88% |

### Class Distribution

| Class | Count |
|---------|---------:|
| Non Defect (0) | 1286 |
| Defect (1) | 66 |

---

# 🏭 Machine Learning Pipeline

```text
Raw Data
    │
    ▼
Missing Value Handling
    │
    ▼
Feature Selection
    │
    ▼
XGBoost Training
    │
    ▼
Probability Prediction
    │
    ▼
Threshold Optimization
    │
    ▼
Final Submission
```

---

# 🧹 Data Preprocessing

## Missing Values

Median Imputation was used for handling missing values.

### Why Median?

- Robust against outliers
- Suitable for industrial process data
- Performed better than mean imputation

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
```

---

# 🎯 Feature Selection Strategy

An initial XGBoost model was trained on all available features.

Feature importance values were extracted and multiple experiments were conducted:

- Top 10 Features
- Top 11 Features
- Top 12 Features
- Top 15 Features
- Top 20 Features
- Recursive Feature Elimination
- Domain-based Feature Engineering

---

# ⭐ Final Selected Features

| Rank | Feature |
|---------|---------|
| 1 | X13 |
| 2 | X35 |
| 3 | X36 |
| 4 | X32 |
| 5 | X14 |
| 6 | X1 |
| 7 | X16 |
| 8 | X49 |
| 9 | X48 |
| 10 | X41 |
| 11 | X21 |
| 12 | X39 |

---

# 📈 Feature Analysis

Statistical analysis revealed strong separation between defective and non-defective coils.

## Important Observations

| Feature | Defect Trend |
|----------|----------|
| X13 | Higher in Defect |
| X32 | Higher in Defect |
| X35 | Lower in Defect |
| X36 | Lower in Defect |
| X39 | Lower in Defect |

---

# 📊 Feature Separation Table

| Feature | Normal Mean | Defect Mean |
|----------|------------:|------------:|
| X13 | 846.16 | 1312.33 |
| X35 | 10.32M | 2.16M |
| X36 | 2376.82 | 378.88 |
| X32 | 15.82 | 19.52 |
| X39 | 162.51 | 157.83 |

These variables contributed most significantly to defect prediction.

---

# ⚖️ Class Imbalance Handling

The dataset is highly imbalanced.

```text
Non Defect = 1286
Defect = 66
```

Class imbalance was handled using:

```python
scale_pos_weight = 19
```

Additional techniques tested:

- Class Weighting
- Focal Loss
- Pseudo Labeling
- Ensemble Learning

The final XGBoost approach achieved the best results.

---

# 🤖 Final Model

## XGBoost Classifier

```python
XGBClassifier(
    n_estimators=511,
    max_depth=3,
    learning_rate=0.03,
    scale_pos_weight=19,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42
)
```

---

# 🎚 Threshold Optimization

Instead of using default predictions, probability outputs were analyzed.

Multiple thresholds were evaluated.

## Best Threshold

```python
0.00095
```

Prediction Rule:

```python
if probability > 0.00095:
    predict = 1
else:
    predict = 0
```

---

# 🧪 Experiments Conducted

| Experiment | Result |
|------------|---------|
| Feature Importance Selection | ✅ |
| Threshold Tuning | ✅ |
| Top-N Features | ✅ |
| X39 Feature Addition | ✅ |
| CatBoost | ❌ |
| Random Forest | ❌ |
| Extra Trees | ❌ |
| Ensemble Models | ❌ |
| Pseudo Labeling | ❌ |
| Interaction Features | ❌ |
| RFE | ❌ |

---

# 🏆 Best Solution

## Configuration

### Features

```text
X13
X35
X36
X32
X14
X1
X16
X49
X48
X41
X21
X39
```

### Model

```python
XGBClassifier(
    n_estimators=511,
    max_depth=3,
    learning_rate=0.03,
    scale_pos_weight=19,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42
)
```

### Threshold

```python
0.00095
```

---

# 📁 Repository Structure

```text
TataSteel-Defect-Prediction/
│
├── README.md
├── final_model.py
├── requirements.txt
├── FINAL_84_63.csv
│
├── notebooks/
│   ├── feature_selection.ipynb
│   ├── threshold_tuning.ipynb
│
└── outputs/
    └── final_submission.csv
```

---

# 📷 Results

## Best Leaderboard Score

| Metric | Value |
|----------|---------|
| Public Score | 84.63 |

---

# 🚀 Future Improvements

Potential future work:

- SHAP-based feature selection
- Automated feature generation
- Domain-specific steel process features
- Advanced anomaly detection
- Hybrid rule-based + ML systems

---

# 🛠 Technologies Used

| Tool | Purpose |
|--------|----------|
| Python | Programming |
| Pandas | Data Processing |
| NumPy | Numerical Computation |
| Scikit-Learn | ML Utilities |
| XGBoost | Classification Model |
| Google Colab | Development Environment |

---

# 📦 Installation

```bash
pip install pandas
pip install numpy
pip install scikit-learn
pip install xgboost
```

or

```bash
pip install -r requirements.txt
```

---

# ▶️ Run

```bash
python final_model.py
```

Output:

```text
FINAL_84_63.csv
```

---

# 👨‍💻 Author

## Kaushal Raj

B.Tech Student | Data Science & Machine Learning Enthusiast

### Areas of Interest

- Machine Learning
- Industrial AI
- Predictive Maintenance
- Data Science
- Railway Technology

GitHub: https://github.com/YOUR_USERNAME

LinkedIn: https://linkedin.com/in/YOUR_PROFILE

---

# ⭐ Acknowledgements

This project was developed as part of the Tata Steel Defect Prediction Challenge and focuses on applying machine learning techniques to industrial manufacturing data for quality prediction and process optimization.

---

## 📜 License

This repository is intended for educational, research, and competition purposes.
