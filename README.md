# HOT-ROLLING-DEFECT-PREDICTION-CHALLENGE
Machine Learning solution for steel defect prediction using XGBoost. Includes feature engineering, class imbalance handling, model optimization, and production-ready prediction pipeline. Developed by Kaushal Raj.

TATA STEEL DEFECT PREDICTION CHALLENGE
======================================

Participant:
Kaushal Raj

Project:
Defect Prediction in Steel Manufacturing Process using Machine Learning

----------------------------------------------------------------------
1. PROBLEM STATEMENT
----------------------------------------------------------------------

The objective of this challenge is to predict whether a steel coil
contains a manufacturing defect (Y=1) or not (Y=0) based on process
parameters collected during production.

The dataset contains multiple numerical process variables (X1 to X49)
and a binary target variable Y.

Y = 0 → Non Defective Coil
Y = 1 → Defective Coil

The challenge involves handling:

- Highly imbalanced data
- Missing values
- Complex relationships between process parameters
- Rare defect occurrences

----------------------------------------------------------------------
2. DATA UNDERSTANDING
----------------------------------------------------------------------

Training Dataset Size:
1352 samples

Class Distribution:

Non Defective (Y=0): 1286
Defective (Y=1): 66

Defect Percentage:
~4.88%

The dataset is highly imbalanced, making defect prediction difficult.

----------------------------------------------------------------------
3. DATA PREPROCESSING
----------------------------------------------------------------------

3.1 Missing Value Handling

Several features contained missing values.

Median Imputation was used because:

- Robust to outliers
- Suitable for skewed industrial data
- Performed better than mean imputation during experiments

Implementation:

SimpleImputer(strategy='median')

3.2 Feature Selection Dataset

All selected features were transformed using the same preprocessing
pipeline for both training and testing datasets.

----------------------------------------------------------------------
4. FEATURE SELECTION APPROACH
----------------------------------------------------------------------

An initial XGBoost model was trained on all available features.

Feature importance values were extracted and ranked.

Multiple experiments were conducted using:

- Top 10 features
- Top 11 features
- Top 12 features
- Top 15 features
- Top 20 features

After extensive testing, the following feature set produced the best
leaderboard score:

Selected Features:

1. X13
2. X35
3. X36
4. X32
5. X14
6. X1
7. X16
8. X49
9. X48
10. X41
11. X21
12. X39

Notably, feature X39 significantly improved performance despite not
being among the highest-ranked features by importance.

----------------------------------------------------------------------
5. EXPLORATORY ANALYSIS
----------------------------------------------------------------------

Statistical analysis revealed clear differences between defective and
non-defective coils.

Important observations:

X13:
- Higher values associated with defects

X35:
- Significantly lower values associated with defects

X36:
- Significantly lower values associated with defects

X32:
- Higher values associated with defects

X39:
- Lower values associated with defects

These features were consistently identified as the most informative.

----------------------------------------------------------------------
6. MODEL DEVELOPMENT
----------------------------------------------------------------------

Several machine learning approaches were evaluated.

Models Tested:

1. XGBoost
2. CatBoost
3. Extra Trees
4. Random Forest Variants
5. Ensemble Approaches
6. Pseudo Labeling Approaches
7. Recursive Feature Elimination (RFE)

Among all tested methods, XGBoost achieved the best performance.

----------------------------------------------------------------------
7. FINAL MODEL
----------------------------------------------------------------------

Algorithm:
XGBoost Classifier

Parameters:

n_estimators = 511
max_depth = 3
learning_rate = 0.03
scale_pos_weight = 19
subsample = 0.9
colsample_bytree = 0.8
random_state = 42

Reasoning:

- Handles non-linear relationships effectively
- Performs well on tabular industrial datasets
- Robust to feature scaling
- Handles imbalanced classification using scale_pos_weight

----------------------------------------------------------------------
8. CLASS IMBALANCE HANDLING
----------------------------------------------------------------------

Since only ~5% of samples belonged to the defect class,
class imbalance was addressed using:

scale_pos_weight = 19

This increased the importance of defect samples during training.

Additional methods tested:

- Focal Loss
- Class Weighting
- Pseudo Labeling
- Ensemble Strategies

However, the final XGBoost approach remained superior.

----------------------------------------------------------------------
9. THRESHOLD OPTIMIZATION
----------------------------------------------------------------------

Instead of directly using default class predictions,
probability outputs were analyzed.

Several threshold values were tested.

Best threshold:

0.00095

Final prediction rule:

If Probability > 0.00095:
    Predict Defect (Y=1)

Else:
    Predict Non Defect (Y=0)

This threshold achieved the best leaderboard performance.

----------------------------------------------------------------------
10. EXPERIMENTS PERFORMED
----------------------------------------------------------------------

The following experiments were conducted:

✓ Feature Importance Based Selection
✓ Threshold Optimization
✓ Top-N Feature Testing
✓ X39 Feature Inclusion
✓ CatBoost Models
✓ Extra Trees Models
✓ Random Forest Style Models
✓ Ensemble Learning
✓ Pseudo Labeling
✓ Interaction Features
✓ Domain Driven Features
✓ Recursive Feature Elimination

The final solution consistently outperformed these alternatives.

----------------------------------------------------------------------
11. FINAL SOLUTION PIPELINE
----------------------------------------------------------------------

Step 1:
Load Training and Test Data

Step 2:
Select Features

[X13, X35, X36, X32, X14, X1,
 X16, X49, X48, X41, X21, X39]

Step 3:
Median Imputation

Step 4:
Train XGBoost Model

Step 5:
Generate Prediction Probabilities

Step 6:
Apply Threshold = 0.00095

Step 7:
Generate Submission File

----------------------------------------------------------------------
12. FILES INCLUDED
----------------------------------------------------------------------

final_model.py
    Main source code

README.txt
    Project documentation

requirements.txt
    Python dependencies

FINAL_84_63.csv
    Final submission file

----------------------------------------------------------------------
13. SOFTWARE AND TOOLS USED
----------------------------------------------------------------------

Programming Language:
Python 3.x

Libraries:

- pandas
- numpy
- scikit-learn
- xgboost

Development Environment:

- Google Colab
- Jupyter Notebook

----------------------------------------------------------------------
14. FINAL RESULT
----------------------------------------------------------------------

Best Public Leaderboard Score:

84.63

Model:

XGBoost + Selected Features + Threshold Optimization

----------------------------------------------------------------------
END OF DOCUMENT
----------------------------------------------------------------------
