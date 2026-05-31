
# ============================================================
# Tata Steel Defect Prediction Challenge
# Final Best Model
# Author: Kaushal Raj
# ============================================================

import pandas as pd

from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# ============================================================
# LOAD DATA
# ============================================================

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# ============================================================
# SELECTED FEATURES
# ============================================================

FEATURES = [

    'X13',
    'X35',
    'X36',
    'X32',
    'X14',

    'X1',
    'X16',
    'X49',
    'X48',
    'X41',

    'X21',
    'X39'
]

# ============================================================
# PREPARE DATA
# ============================================================

X_train = train[FEATURES]

y_train = train['Y']

X_test = test[FEATURES]

# ============================================================
# HANDLE MISSING VALUES
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(
    X_train
)

X_test = imputer.transform(
    X_test
)

# ============================================================
# MODEL
# ============================================================

model = XGBClassifier(

    n_estimators=511,

    max_depth=3,

    learning_rate=0.03,

    scale_pos_weight=19,

    subsample=0.9,

    colsample_bytree=0.8,

    random_state=42,

    eval_metric="logloss"
)

# ============================================================
# TRAIN MODEL
# ============================================================

print("Training Model...")

model.fit(
    X_train,
    y_train
)

print("Training Complete")

# ============================================================
# PREDICT PROBABILITIES
# ============================================================

probabilities = model.predict_proba(
    X_test
)[:, 1]

# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

THRESHOLD = 0.00095

predictions = (
    probabilities > THRESHOLD
).astype(int)

# ============================================================
# CREATE SUBMISSION
# ============================================================

submission = pd.DataFrame({

    "CoilID": test["CoilID"],

    "Y": predictions
})

# ============================================================
# SAVE FILE
# ============================================================

OUTPUT_FILE = "FINAL_84_63.csv"

submission.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("\nSubmission Saved Successfully")

print(f"\nOutput File: {OUTPUT_FILE}")

print("\nPrediction Distribution:")

print(
    submission["Y"].value_counts()
)

print("\nFirst 5 Predictions:")

print(
    submission.head()
)

print("\nDone.")
