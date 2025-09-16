# Download NLTK data once
import nltk

if __name__ == "__main__":
    nltk.download("punkt")        # sentence tokenizer
    nltk.download("punkt_tab")    # NEW: punkt tables (needed in latest NLTK)
    nltk.download("stopwords")
    nltk.download("wordnet")
    print("Downloaded punkt, punkt_tab, stopwords, wordnet.")
