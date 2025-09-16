
import os, json, joblib, pandas as pd
from sklearn.metrics import classification_report
from .preprocess import preprocess_for_vectorizer
from .config import MODEL_DIR, REPORTS_DIR

def main():
    pipe = joblib.load(os.path.join(MODEL_DIR, "model_pipeline.joblib"))
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "synthetic_notes.csv"))
    df["text_clean"] = df["note_text"].apply(preprocess_for_vectorizer)
    y_true = df["label"]
    y_pred = pipe.predict(df["text_clean"])
    report = classification_report(y_true, y_pred, output_dict=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "eval_full_dataset.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("Saved reports/eval_full_dataset.json")

if __name__ == "__main__":
    main()
