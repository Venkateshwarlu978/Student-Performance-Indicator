import os
import sys
import dill
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- Helper to save object ---
def save_obj(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        dill.dump(obj, f)

# --- Try to load CSV from common locations, else synthesize data ---
def load_dataset():
    candidates = [
        "data/students_performance.csv",
        "data/StudentsPerformance.csv",
        "StudentsPerformance.csv",
        "data/student_scores.csv",
        "student_scores.csv",
    ]

    for p in candidates:
        if os.path.exists(p):
            print(f"Loading dataset from {p}")
            return pd.read_csv(p)

    # fallback synthetic dataset (small)
    print("No input CSV found. Creating a synthetic dataset for training.")
    rng = np.random.default_rng(42)
    n = 1500
    genders = rng.choice(["male", "female"], size=n)
    groups = rng.choice(["group A", "group B", "group C", "group D", "group E"], size=n)
    parents = rng.choice(["bachelor's degree", "some college", "master's degree", "associate's degree", "high school"], size=n)
    lunches = rng.choice(["standard", "free/reduced"], size=n)
    prep = rng.choice(["none", "completed"], size=n)

    reading = rng.integers(30, 100, size=n)
    writing = rng.integers(25, 100, size=n)
    # target: simulate math score correlated with reading/writing + noise
    math = (reading * 0.45 + writing * 0.45 + rng.normal(0, 6, size=n)).round().astype(int)
    math = np.clip(math, 0, 100)

    df = pd.DataFrame({
        "gender": genders,
        "race_ethnicity": groups,
        "parental_level_of_education": parents,
        "lunch": lunches,
        "test_preparation_course": prep,
        "reading_score": reading,
        "writing_score": writing,
        "math_score": math
    })
    return df

def prepare_and_train(df):
    # Ensure required columns exist (normalize names if necessary)
    # map common column names that datasets might use
    colmap = {}
    lower_cols = {c.lower(): c for c in df.columns}
    # common variants
    if "reading score" in lower_cols:
        colmap[lower_cols["reading score"]] = "reading_score"
    if "writing score" in lower_cols:
        colmap[lower_cols["writing score"]] = "writing_score"
    if "math score" in lower_cols:
        colmap[lower_cols["math score"]] = "math_score"
    if "race/ethnicity" in lower_cols:
        colmap[lower_cols["race/ethnicity"]] = "race_ethnicity"
    if "parental level of education" in lower_cols:
        colmap[lower_cols["parental level of education"]] = "parental_level_of_education"
    if "test preparation course" in lower_cols:
        colmap[lower_cols["test preparation course"]] = "test_preparation_course"

    if colmap:
        df = df.rename(columns=colmap)

    # If no explicit math target present, create one from reading+writing
    if "math_score" not in df.columns:
        df["math_score"] = ((df["reading_score"].astype(float) + df["writing_score"].astype(float)) / 2 + np.random.normal(0, 5, size=len(df))).round().astype(int)
        df["math_score"] = df["math_score"].clip(0, 100)

    # select feature columns expected by your app
    cat_features = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
    num_features = ["reading_score", "writing_score"]

    # Ensure columns exist; if not, try to fill with defaults
    for c in cat_features:
        if c not in df.columns:
            print(f"Warning: column '{c}' not found in dataset. Filling with default values.")
            df[c] = "unknown"
    for c in num_features:
        if c not in df.columns:
            print(f"Error: numeric column '{c}' not found — creating zeros.")
            df[c] = 0

    X = df[cat_features + num_features].copy()
    y = df["math_score"].copy()

    # simple train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessor: onehot for categorical, scaler for numeric
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)
    num_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, cat_features),
            ("num", num_transformer, num_features),
        ],
        remainder="drop",
    )

    # Model pipeline (preprocessor saved separately so Flask can use it like your code expects)
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit preprocessor and transform training data for model
    print("Fitting preprocessor...")
    preprocessor.fit(X_train)

    print("Transforming training data...")
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    print("Training model...")
    model.fit(X_train_trans, y_train)

    # Evaluate
    preds = model.predict(X_test_trans)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"Validation RMSE: {rmse:.3f}")

    # Save preprocessor and model to artifacts/
    artifacts_dir = os.path.join(os.getcwd(), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    preprocessor_path = os.path.join(artifacts_dir, "preprocessor.pkl")
    model_path = os.path.join(artifacts_dir, "model.pkl")

    print(f"Saving preprocessor to {preprocessor_path}")
    save_obj(preprocessor, preprocessor_path)

    print(f"Saving model to {model_path}")
    save_obj(model, model_path)

    print("Training complete and artifacts saved.")
    return preprocessor_path, model_path

def main():
    try:
        df = load_dataset()
        preproc_path, model_path = prepare_and_train(df)
    except Exception as e:
        print("Training failed:", e)
        raise

if __name__ == "__main__":
    main()
