# FAQ Chatbot (NLP)

A chatbot that answers questions from a FAQ dataset using NLP preprocessing
(NLTK) and TF-IDF + cosine similarity (scikit-learn), with a Streamlit chat UI.

## How it works
1. **Collect FAQs** – `faq.json` holds question/answer pairs (22 college FAQs) plus optional `keywords` for synonyms.
2. **Preprocess** – `preprocess.py`: lowercase, remove URLs/punctuation, tokenize, remove stopwords (keeping "no"/"not"), lemmatize.
3. **Match** – `chatbot.py`: `TfidfVectorizer` (unigrams + bigrams) is fitted on the FAQ questions; the user query is transformed the same way and compared using cosine similarity.
4. **Respond** – the answer of the best match is returned. If similarity is below the threshold (0.20), the bot says it didn't understand and suggests related questions.
5. **UI** – `app.py`: Streamlit chat interface with chat history, sample questions and an optional "match details" toggle.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py          # chat UI
python chatbot.py             # or the command-line version
```
NLTK data (punkt, stopwords, wordnet) downloads automatically on first run.

## Use your own topic
Edit `faq.json` (keep the `question`, `answer`, `keywords` fields). No code changes needed.
Tune `THRESHOLD` in `chatbot.py` if the bot is too strict or too lenient.

## Possible improvements
- Replace TF-IDF with `sentence-transformers` embeddings for better paraphrase matching.
- Log unanswered questions to find missing FAQs.
