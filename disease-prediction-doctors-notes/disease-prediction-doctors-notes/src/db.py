
import argparse
import json
import os
import pandas as pd
from typing import List, Tuple, Optional, Any
from .config import DB_BACKEND, DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, SQLITE_PATH

def _mysql_conn():
    import mysql.connector  # local import so project works even if mysql isn't installed yet
    return mysql.connector.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
    )

def _sqlite_conn():
    import sqlite3
    conn = sqlite3.connect(SQLITE_PATH)
    # emulate JSON column
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def get_conn():
    if DB_BACKEND == "sqlite":
        return _sqlite_conn()
    return _mysql_conn()

def init_sqlite_schema():
    if DB_BACKEND != "sqlite":
        return
    import sqlite3
    with get_conn() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS notes (
            note_id INTEGER PRIMARY KEY,
            note_text TEXT NOT NULL,
            label TEXT NOT NULL
        );""")
        conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
            pred_id INTEGER PRIMARY KEY AUTOINCREMENT,
            note_id INTEGER NOT NULL,
            predicted_label TEXT NOT NULL,
            proba TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );""")
        conn.commit()

def create_tables_if_needed():
    # For MySQL, assume schema applied via sql/schema.sql. For SQLite, create programmatically.
    if DB_BACKEND == "sqlite":
        init_sqlite_schema()

def insert_notes(df: pd.DataFrame):
    create_tables_if_needed()
    with get_conn() as conn:
        cur = conn.cursor()
        if DB_BACKEND == "mysql":
            sql = "INSERT INTO notes (note_id, note_text, label) VALUES (%s, %s, %s)"
        else:
            sql = "INSERT OR REPLACE INTO notes (note_id, note_text, label) VALUES (?, ?, ?)"
        for _, row in df.iterrows():
            cur.execute(sql, (int(row["note_id"]), str(row["note_text"]), str(row["label"])))
        conn.commit()

def fetch_notes(limit: Optional[int] = None) -> List[Tuple[int, str, str]]:
    create_tables_if_needed()
    with get_conn() as conn:
        cur = conn.cursor()
        q = "SELECT note_id, note_text, label FROM notes"
        if limit:
            if DB_BACKEND == "mysql":
                q += f" LIMIT {int(limit)}"
            else:
                q += f" LIMIT {int(limit)}"
        cur.execute(q)
        rows = cur.fetchall()
    return rows

def save_prediction(note_id: int, predicted_label: str, proba: Optional[dict] = None):
    create_tables_if_needed()
    with get_conn() as conn:
        cur = conn.cursor()
        if DB_BACKEND == "mysql":
            sql = "INSERT INTO predictions (note_id, predicted_label, proba) VALUES (%s, %s, %s)"
            cur.execute(sql, (note_id, predicted_label, json.dumps(proba) if proba else None))
        else:
            sql = "INSERT INTO predictions (note_id, predicted_label, proba) VALUES (?, ?, ?)"
            cur.execute(sql, (note_id, predicted_label, json.dumps(proba) if proba else None))
        conn.commit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_csv", type=str, help="Path to CSV to insert into notes table")
    args = parser.parse_args()
    if args.load_csv:
        df = pd.read_csv(args.load_csv)
        insert_notes(df)
        print(f"Loaded {len(df)} rows into 'notes'.")
    else:
        print("Nothing to do. Use --load_csv data/synthetic_notes.csv")
