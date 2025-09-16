
import re
from typing import List
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_STOPWORDS = set(stopwords.words("english"))
_LEM = WordNetLemmatizer()

def clean_text(text: str) -> str:
    # Lowercase
    text = text.lower()
    # Remove non-alphabetic characters, keep spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_and_lemmatize(text: str) -> List[str]:
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    tokens = [_LEM.lemmatize(t) for t in tokens]
    return tokens

def preprocess_for_vectorizer(text: str) -> str:
    # Return a space-joined string of lemmas (works well with TF-IDF)
    cleaned = clean_text(text)
    return " ".join(tokenize_and_lemmatize(cleaned))
