
import argparse
from .pipeline import predict_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disease prediction from a single clinical note (CLI)")
    parser.add_argument("--text", required=True, help="Doctor's note text to classify")
    args = parser.parse_args()
    label, proba = predict_text(args.text)
    print("Predicted Label:", label)
    if proba:
        print("Probabilities:")
        for k, v in sorted(proba.items(), key=lambda kv: -kv[1]):
            print(f"  {k}: {v:.3f}")
