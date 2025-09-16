
# Disease Prediction from Doctor’s Notes (NLP + ML)

An end‑to‑end, **teaching‑oriented** project that turns unstructured clinical notes into disease predictions using classical NLP (TF‑IDF) + ML (Logistic Regression / SVM), integrated with **MySQL** for data storage and **Git/GitHub** workflows.

## 📦 What’s Inside
- `data/synthetic_notes.csv` — a small synthetic dataset to start immediately.
- `sql/schema.sql` — MySQL tables for notes and predictions.
- `src/preprocess.py` — text cleaning, tokenization, lemmatization.
- `src/db.py` — MySQL helpers (create tables, insert/fetch notes, save predictions).
- `src/train.py` — train baseline models (TF‑IDF + LogisticRegression/SVM).
- `src/evaluate.py` — evaluate models with precision/recall/F1.
- `src/pipeline.py` — end‑to‑end: fetch -> preprocess -> predict -> store.
- `src/cli.py` — quick CLI for ad‑hoc predictions.
- `src/config.py` — central place for environment variables.
- `src/prepare_nltk.py` — downloads NLTK resources once.
- `models/` — saved vectorizer + model (after training).
- `reports/` — evaluation reports saved here.

> **Tech**: Linux, Python, Git/GitHub, MySQL, pandas, scikit‑learn, NLTK, joblib.

---

## 🧰 Setup

1) **Python env**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.prepare_nltk  # download NLTK data once
```

2) **MySQL**
- Start a local MySQL server.
- Create a DB, e.g. `clinical_nlp`.
- Update **env vars** (see below) or edit `src/config.py`.

3) **Environment variables**
Create a `.env` file (or export in shell):
```
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=yourpassword
DB_NAME=clinical_nlp
# Optional: set to 'sqlite' for local/offline testing
DB_BACKEND=mysql
SQLITE_PATH=./local.db
```

4) **Load schema**
```bash
mysql -u $DB_USER -p$DB_PASSWORD -h $DB_HOST -P $DB_PORT $DB_NAME < sql/schema.sql
```

5) **Load synthetic data into MySQL**
```bash
python -m src.db --load_csv data/synthetic_notes.csv
```

6) **Train & Evaluate**
```bash
python -m src.train
python -m src.evaluate
```

7) **End‑to‑End Pipeline (DB -> Predict -> Store)**
```bash
python -m src.pipeline
```

8) **Ad‑hoc Prediction from CLI**
```bash
python -m src.cli --text "Patient reports fever, cough, body aches for 3 days."
```

---

## 🧪 Dataset

`data/synthetic_notes.csv` has three columns:
- `note_id` — integer ID
- `note_text` — clinical note (free text)
- `label` — target diagnosis (class)

This is a **toy** dataset for learning. Replace with your curated dataset once ready (ensure the same columns).

---

## 🧠 Models

- Baseline: **TF‑IDF + Logistic Regression**
- Alternative: **Linear SVM (SGDClassifier hinge)**

Saved to `models/vectorizer.joblib` and `models/model.joblib`.

---

## 🗃️ MySQL Schema (summary)

- `notes(note_id INT PK, note_text TEXT, label VARCHAR(64))`
- `predictions(pred_id INT PK AUTO, note_id INT FK, predicted_label VARCHAR(64), proba JSON, created_at TIMESTAMP)`

---

## 🔁 Typical Workflow (Phase‑2 Project Days)

1. **Day 6–7**: Ingest/clean/EDA, load MySQL.
2. **Day 8**: Build baseline TF‑IDF + LR model.
3. **Day 9–10**: Improve model (SVM, tuning), log results.
4. **Day 11**: Wire pipeline DB -> preprocess -> predict -> store.
5. **Day 12–13**: Tests, CLI, docs, error handling.
6. **Day 14**: Final demo, README polish, repo hygiene.

---

## 📜 License
MIT — free to use for learning and internal projects.
