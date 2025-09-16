import os
import joblib
from .preprocess import preprocess_for_vectorizer
from .config import MODEL_DIR
from . import db as dbmod


# Function: Make a prediction for a single text
def predict_text(text):
    # Load the trained model
    model_path = os.path.join(MODEL_DIR, "model_pipeline.joblib")
    pipeline = joblib.load(model_path)

    # Clean the text
    cleaned_text = preprocess_for_vectorizer(text)

    # Predict the label
    prediction = pipeline.predict([cleaned_text])[0]

    # Try to get prediction probabilities
    probabilities = None
    try:
        proba_array = pipeline.predict_proba([cleaned_text])[0]
        labels = list(pipeline.classes_)

        # Build dictionary like {"Positive": 0.8, "Negative": 0.2}
        probabilities = {}
        for i in range(len(labels)):
            probabilities[labels[i]] = float(proba_array[i])
    except:
        probabilities = None

    return prediction, probabilities


# Function: Run the pipeline on notes from the database
def run_pipeline(limit=None):
    rows = dbmod.fetch_notes(limit=limit)

    for note_id, note_text, _ in rows:
        pred, proba = predict_text(note_text)
        dbmod.save_prediction(note_id, pred, proba)

    print("Predicted and saved", len(rows), "rows.")


# Run only if this file is executed directly
if __name__ == "__main__":
    run_pipeline()
