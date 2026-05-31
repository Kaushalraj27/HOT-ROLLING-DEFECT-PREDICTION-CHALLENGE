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

<img width="560" height="463" alt="class_distribution" src="https://github.com/user-attachments/assets/de8be1f7-2537-4d69-8fde-35431ade5ec0" />


---

# 🏭 Machine Learning Pipeline

The complete workflow followed in this project.

<img width="1360" height="840" alt="ml_pipeline_horizontal" src="https://github.com/user-attachments/assets/370a35fc-41ba-46ff-9c26-f21b4448d0fc" />


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

<img width="831" height="528" alt="feature_importance" src="https://github.com/user-attachments/assets/782dab46-d6fd-4577-9808-f1e43ac7917c" />


---

# 🔬 Feature Analysis

## X13 Distribution

Defective coils generally show significantly higher X13 values.

<img width="562" height="455" alt="x13_distribution" src="https://github.com/user-attachments/assets/1f2a0351-4cbc-45c7-9d82-c36dcbe88866" />

---

## X35 Distribution

Defective coils generally show much lower X35 values.

<img width="571" height="455" alt="x35_distribution" src="https://github.com/user-attachments/assets/927615c6-2c5d-4f64-a8f4-2f94d1d2be22" />


---

## X36 Distribution

Defective coils generally show lower X36 values.

<img width="571" height="455" alt="x36_distribution" src="https://github.com/user-attachments/assets/c0fb4345-ec3a-4e33-bfcc-2743f91ab473" />


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

GitHub: [https://github.com/Kaushalraj27]

LinkedIn: [https://linkedin.com/in/kaushal-raj21]

---

# ⭐ Acknowledgements

Developed for the Tata Steel Defect Prediction Challenge.

Focused on industrial process analytics, defect prediction and machine learning based quality assurance.

---

# 📜 License

Educational and Research Use.
