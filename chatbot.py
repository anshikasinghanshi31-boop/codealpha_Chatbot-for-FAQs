"""FAQ chatbot engine: TF-IDF + cosine similarity. Run this file for a CLI chat."""
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess

FAQ_PATH = Path(__file__).parent / "faq.json"
THRESHOLD = 0.20   # below this similarity we say "I don't know"

GREETINGS = {"hi", "hello", "hey", "hii", "good morning", "good evening"}
GOODBYES = {"bye", "goodbye", "exit", "quit", "thanks", "thank you"}


class FAQChatbot:
    def __init__(self, faq_path=FAQ_PATH, threshold=THRESHOLD):
        with open(faq_path, encoding="utf-8") as f:
            self.faqs = json.load(f)
        self.threshold = threshold

        # Index = question + keywords, so synonyms in "keywords" also match.
        corpus = [preprocess(f"{x['question']} {x.get('keywords', '')}") for x in self.faqs]
        # Unigrams + bigrams give better matching for short questions.
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.faq_matrix = self.vectorizer.fit_transform(corpus)

    def top_matches(self, query, k=3):
        """Return [(index, score), ...] of the k most similar FAQs."""
        q_vec = self.vectorizer.transform([preprocess(query)])
        scores = cosine_similarity(q_vec, self.faq_matrix).ravel()
        best = scores.argsort()[::-1][:k]
        return [(int(i), float(scores[i])) for i in best]

    def get_response(self, query):
        """Return (answer, matched_question, score, suggestions)."""
        cleaned = query.strip().lower()
        if cleaned in GREETINGS:
            return "Hello! Ask me anything about the college.", None, 1.0, []
        if cleaned in GOODBYES:
            return "Thanks for chatting. Goodbye!", None, 1.0, []

        matches = self.top_matches(query)
        idx, score = matches[0]
        if score < self.threshold:
            suggestions = [self.faqs[i]["question"] for i, s in matches if s > 0]
            return ("Sorry, I couldn't find a good answer to that. "
                    "Try rephrasing your question."), None, score, suggestions
        suggestions = [self.faqs[i]["question"] for i, s in matches[1:] if s > 0]
        return self.faqs[idx]["answer"], self.faqs[idx]["question"], score, suggestions


def main():
    bot = FAQChatbot()
    print("College FAQ Bot (type 'quit' to exit)\n")
    while True:
        query = input("You: ").strip()
        if not query:
            continue
        answer, matched, score, suggestions = bot.get_response(query)
        print(f"Bot: {answer}")
        if matched:
            print(f"     [matched: '{matched}' | similarity: {score:.2f}]")
        if suggestions:
            print("     Related questions: " + " | ".join(suggestions))
        if query.lower() in GOODBYES:
            break


if __name__ == "__main__":
    main()
