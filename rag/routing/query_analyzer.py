import re
import numpy as np
import ollama
from sklearn.metrics.pairwise import cosine_similarity


class QueryAnalyzer:

    def __init__(self):

        self.embedding_model = "nomic-embed-text"
        self.llm_model = "iKhalid/ALLaM:7b"

        # threshold for semantic confidence
        self.semantic_threshold = 0.75

        # intent examples (minimal seeds)
        self.intents = {
            "document_lookup": [
                "revenue in 2020",
                "financial report numbers",
                "income statement value"
            ],
            "market_data": [
                "tesla stock price",
                "bitcoin price",
                "gold price today"
            ],
            "general_finance": [
                "what is diversification",
                "what is investing",
                "define portfolio"
            ],
            "analysis": [
                "should i invest in tesla",
                "is diversification good",
                "portfolio strategy advice"
            ],
            "forecast": [
                "predict tesla price",
                "future stock price",
                "market forecast"
            ]
        }

        # precompute embeddings
        self.intent_embeddings = {}

        for intent, examples in self.intents.items():

            embeddings = []

            for text in examples:

                emb = ollama.embeddings(
                    model=self.embedding_model,
                    prompt=text
                )["embedding"]

                embeddings.append(emb)

            self.intent_embeddings[intent] = np.array(embeddings)

    # -----------------------------
    # Rule Layer
    # -----------------------------
    def rule_layer(self, question):

        q = question.lower()

        if re.search(r"price|stock|ticker|usd|btc|gold|oil", q):
            return "market_data", 0.95

        if re.search(r"revenue|income|financial statement|report", q):
            return "document_lookup", 0.95

        if re.search(r"predict|forecast|future price", q):
            return "forecast", 0.95

        return None, 0

    # -----------------------------
    # Embedding Layer
    # -----------------------------
    def embedding_layer(self, question):

        q_emb = ollama.embeddings(
            model=self.embedding_model,
            prompt=question
        )["embedding"]

        q_emb = np.array(q_emb).reshape(1, -1)

        best_intent = None
        best_score = 0

        for intent, emb in self.intent_embeddings.items():

            score = cosine_similarity(q_emb, emb).max()

            if score > best_score:

                best_score = score
                best_intent = intent

        return best_intent, best_score

    # -----------------------------
    # LLM Fallback
    # -----------------------------
    def llm_layer(self, question):

        prompt = f"""
Classify the following finance question into ONE category.

Categories:
document_lookup
market_data
general_finance
analysis
forecast

Return ONLY the category name.

Question:
{question}
"""

        response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"].strip()

    # -----------------------------
    # Main Analyzer
    # -----------------------------
    def analyze(self, question):

        # Rule layer
        category, confidence = self.rule_layer(question)

        if confidence > 0.9:

            print("[Analyzer] rule match")

            return category

        # Embedding layer
        intent, score = self.embedding_layer(question)

        print(f"[Analyzer] semantic score: {round(score,3)}")

        if score > self.semantic_threshold:

            print("[Analyzer] semantic match")

            return intent

        # LLM fallback
        print("[Analyzer] fallback to LLM")

        intent = self.llm_layer(question)

        return intent