
CREATE TABLE IF NOT EXISTS notes (
  note_id INT PRIMARY KEY,
  note_text TEXT NOT NULL,
  label VARCHAR(64) NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
  pred_id INT AUTO_INCREMENT PRIMARY KEY,
  note_id INT NOT NULL,
  predicted_label VARCHAR(64) NOT NULL,
  proba JSON NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (note_id) REFERENCES notes(note_id)
);
