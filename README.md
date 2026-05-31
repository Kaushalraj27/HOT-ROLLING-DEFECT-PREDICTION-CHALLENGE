# 🔍 Tata Steel Defect Prediction Challenge

> Machine Learning solution for predicting steel manufacturing defects using XGBoost, feature selection, threshold optimization and imbalance handling.

**Developer:** Kaushal Raj

![Python](https://img.shields.io/badge/Python-3.x-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)
![Score](https://img.shields.io/badge/Best%20Score-84.63-green)

---

# 📌 Problem Statement

Predict whether a steel coil contains manufacturing defects using process parameters collected during production.

---

# 📊 Dataset Overview

| Metric | Value |
|----------|----------|
| Training Samples | 1352 |
| Test Samples | 339 |
| Features | 49 |
| Defect Samples | 66 |
| Non Defect Samples | 1286 |
| Defect Ratio | 4.88% |

---

# 📊 Class Distribution

The dataset is highly imbalanced.

![Class Distribution](images/class_distribution.png)

---

# 🏭 Machine Learning Pipeline

The complete workflow followed in this project.

![Pipeline](images/model_pipeline.png)

---

# 🧹 Data Preprocessing

### Missing Value Handling

Median Imputation was used because:

- Robust to outliers
- Performs well on industrial process data
- Better than mean imputation in experiments

```python
SimpleImputer(strategy="median")
```

---

# 🎯 Feature Selection

Initial XGBoost feature importance was used.

After extensive experimentation the following features produced the best result:

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

# 📈 Feature Importance

Most influential variables selected by the feature model.

![Feature Importance](images/feature_importance.png)

---

# 🔬 Feature Analysis

## X13 Distribution

Defective coils generally show significantly higher X13 values.

![X13 Distribution](images/x13_distribution.png)

---

## X35 Distribution

Defective coils generally show much lower X35 values.

![X35 Distribution](images/x35_distribution.png)

---

## X36 Distribution

Defective coils generally show lower X36 values.

![X36 Distribution](images/x36_distribution.png)

---

# ⚖️ Class Imbalance Handling

The dataset contains only:

```text
66 Defect Samples
1286 Non Defect Samples
```

Class imbalance handled using:

```python
scale_pos_weight = 19
```

---

# 🤖 Final Model

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

Several thresholds were tested.

Best threshold:

```python
0.00095
```

Prediction rule:

```python
if probability > 0.00095:
    prediction = 1
else:
    prediction = 0
```

---

# 🧪 Experiments Conducted

| Experiment | Status |
|------------|---------|
| Feature Importance Selection | ✅ |
| Threshold Tuning | ✅ |
| Top Feature Search | ✅ |
| X39 Feature Inclusion | ✅ |
| CatBoost | ❌ |
| Random Forest | ❌ |
| Extra Trees | ❌ |
| Ensemble Models | ❌ |
| Pseudo Labeling | ❌ |
| RFE | ❌ |
| Interaction Features | ❌ |

---

# 🏆 Best Performing Configuration

## Features

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

## Model

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

## Threshold

```python
0.00095
```

## Best Leaderboard Score

```text
84.63
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
├── images/
│   ├── class_distribution.png
│   ├── feature_importance.png
│   ├── x13_distribution.png
│   ├── x35_distribution.png
│   ├── x36_distribution.png
│   └── model_pipeline.png
│
└── outputs/
```

---

# 🛠 Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Programming |
| Pandas | Data Processing |
| NumPy | Numerical Computation |
| Scikit-Learn | Preprocessing |
| XGBoost | Classification |
| Google Colab | Development |

---

# 🚀 Installation

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
- Data Science
- Industrial AI
- Predictive Maintenance
- Railway Technology

GitHub: [https://github.com/YOUR_USERNAME](https://github.com/Kaushalraj27/)

LinkedIn: [https://linkedin.com/in/YOUR_LINKEDIN](https://www.linkedin.com/in/kaushal-raj21/)

---

# ⭐ Acknowledgements

Developed for the Tata Steel Defect Prediction Challenge.

Focused on industrial process analytics, defect prediction and machine learning based quality assurance.

---

# 📜 License

Educational and Research Use.
