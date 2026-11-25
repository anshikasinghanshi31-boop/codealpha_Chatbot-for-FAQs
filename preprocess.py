"""Text preprocessing with NLTK: lowercase -> clean -> tokenize -> stopwords -> lemmatize."""
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def _ensure_nltk_data():
    """Download required NLTK resources on first run."""
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


_ensure_nltk_data()

# Keep negations: "not" / "no" can change the meaning of a question.
_STOPWORDS = set(stopwords.words("english")) - {"no", "not"}
_LEMMATIZER = WordNetLemmatizer()


def preprocess(text: str) -> str:
    """Return a cleaned, lemmatized version of `text` as a single string."""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)                        # remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _STOPWORDS]
    tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)
