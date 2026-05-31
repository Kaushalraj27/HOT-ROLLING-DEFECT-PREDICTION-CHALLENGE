# Tata Steel Defect Prediction Challenge
## Solution Approach

### Author
Kaushal Raj

---

# 1. Objective

The objective of this challenge is to predict whether a steel coil is defective or non-defective using process parameters collected during manufacturing.

The problem is a binary classification problem:

- Y = 0 → Non Defective Coil
- Y = 1 → Defective Coil

---

# 2. Dataset Analysis

Training Data:

| Metric | Value |
|----------|----------|
| Samples | 1352 |
| Features | 49 |
| Defective Coils | 66 |
| Non Defective Coils | 1286 |

Defect Ratio:

```text
66 / 1352 ≈ 4.88%
```

The dataset is highly imbalanced.

---

# 3. Data Preprocessing

### Missing Value Handling

Several features contained missing values.

Median imputation was used because:

- Robust against outliers
- Suitable for industrial process data
- Produced better results than mean imputation

```python
SimpleImputer(strategy="median")
```

---

# 4. Feature Selection

An initial XGBoost model was trained using all available features.

Feature importance scores were extracted.

Multiple feature subsets were evaluated.

Final selected features:

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

Feature X39 provided a significant improvement despite not being among the highest-ranked importance features.

---

# 5. Exploratory Analysis

Statistical analysis showed strong differences between defective and non-defective coils.

### Key Observations

| Feature | Defect Behaviour |
|----------|----------|
| X13 | Higher |
| X32 | Higher |
| X35 | Lower |
| X36 | Lower |
| X39 | Lower |

These features became the primary drivers of the final model.

---

# 6. Model Development

Several machine learning algorithms were evaluated:

### Tested Models

- XGBoost
- CatBoost
- Extra Trees
- Random Forest Variants
- Ensemble Methods
- Pseudo Labeling
- Recursive Feature Elimination (RFE)

Among all tested approaches, XGBoost consistently produced the best performance.

---

# 7. Handling Class Imbalance

The dataset contains very few defect samples.

Class imbalance was handled using:

```python
scale_pos_weight = 19
```

This increased the importance of defect samples during model training.

---

# 8. Final Model

Algorithm:

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

Reasoning:

- Strong performance on tabular data
- Handles nonlinear relationships
- Robust to feature scaling
- Effective for imbalanced classification

---

# 9. Threshold Optimization

Instead of using default class predictions, probability outputs were analyzed.

Multiple thresholds were tested.

Best threshold:

```python
0.00095
```

Final prediction rule:

```python
if probability > 0.00095:
    predict = 1
else:
    predict = 0
```

---

# 10. Experiments Conducted

| Experiment | Result |
|------------|---------|
| Feature Importance Selection | Successful |
| Threshold Optimization | Successful |
| Top Feature Search | Successful |
| X39 Inclusion | Successful |
| CatBoost | No Improvement |
| Random Forest | No Improvement |
| Extra Trees | No Improvement |
| Ensemble Learning | No Improvement |
| Pseudo Labeling | No Improvement |
| Interaction Features | No Improvement |
| Domain Features | No Improvement |
| RFE | No Improvement |

---

# 11. Final Pipeline

```text
Raw Data
    ↓
Median Imputation
    ↓
Feature Selection
    ↓
XGBoost Training
    ↓
Probability Prediction
    ↓
Threshold Optimization
    ↓
Final Submission
```

---

# 12. Result

Best Public Leaderboard Score:

```text
84.63
```

Final Solution:

- Selected Feature Set
- XGBoost Classifier
- Median Imputation
- Threshold Optimization

---

# 13. Conclusion

The final solution successfully combines feature selection, class imbalance handling, and XGBoost-based classification to predict manufacturing defects in steel coils.

Extensive experimentation was performed to evaluate alternative models and feature engineering strategies. The selected XGBoost configuration provided the most reliable and highest-performing solution among all tested approaches.
