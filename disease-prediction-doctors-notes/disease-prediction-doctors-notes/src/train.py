import os
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

from .preprocess import preprocess_for_vectorizer
from .config import MODEL_DIR, REPORTS_DIR

# You can switch between LogisticRegression and Linear SVM (SGDClassifier) below
USE_SVM = False  # set True to use SVM


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if not {"note_text", "label"}.issubset(df.columns):
        raise ValueError("CSV must have 'note_text' and 'label' columns.")
    return df


def build_pipeline():
    vectorizer = TfidfVectorizer(
        preprocessor=None,  # we pre-clean below
        tokenizer=None,     # we pass cleaned-space-joined tokens
        lowercase=False,    # we already lowercase
        ngram_range=(1, 2),
        max_features=10000
    )
    if USE_SVM:
        base_clf = SGDClassifier(loss="hinge", random_state=42, max_iter=1000)
        clf = CalibratedClassifierCV(base_estimator=base_clf, cv=3)
    else:
        clf = LogisticRegression(max_iter=200, n_jobs=None)
    pipe = Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf),
    ])
    return pipe


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Load dataset
    df = load_dataset(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "synthetic_notes.csv"))

    # Pre-clean text for TF-IDF
    df["text_clean"] = df["note_text"].apply(preprocess_for_vectorizer)

    # ✅ Per-class split (Option 2) → ensures every disease appears in train & test
    train_idx, test_idx = [], []
    for label in df["label"].unique():
        subset = df[df["label"] == label]
        tr, te = train_test_split(subset, test_size=1, random_state=42)  # 1 test per class
        train_idx.extend(tr.index)
        test_idx.extend(te.index)

    X_train, X_test = df.loc[train_idx, "text_clean"], df.loc[test_idx, "text_clean"]
    y_train, y_test = df.loc[train_idx, "label"], df.loc[test_idx, "label"]

    # Build and train model
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    try:
        y_proba = pipe.predict_proba(X_test)
    except Exception:
        y_proba = None

    # report = classification_report(y_test, y_pred, output_dict=True)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    with open(os.path.join(REPORTS_DIR, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("Saved reports/classification_report.json")

    # Save model
    joblib.dump(pipe, os.path.join(MODEL_DIR, "model_pipeline.joblib"))
    print("Saved models/model_pipeline.joblib")


if __name__ == "__main__":
    main()
