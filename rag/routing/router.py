from rag.routing.query_analyzer import QueryAnalyzer
from rag.rag_engine import RAGEngine
from rag.llm.llm_engine import LLMEngine
from rag.online.asset_extractor import AssetExtractor
from rag.online.market_fetcher import MarketFetcher
from rag.online.market_verifier import MarketVerifier


class Router:

    def __init__(self):

        self.analyzer = QueryAnalyzer()
        self.rag = RAGEngine()
        self.llm = LLMEngine()

        # Online market layer
        self.asset_extractor = AssetExtractor()
        self.market_fetcher = MarketFetcher()
        self.market_verifier = MarketVerifier()

        # Forecast model placeholder
        self.forecast_model = None

    # -----------------------------------
    # Confidence calculation for RAG
    # -----------------------------------
    def calculate_confidence(self, distances):

        if not distances:
            return 70.0

        similarities = [1 - d for d in distances]

        avg_similarity = sum(similarities) / len(similarities)

        strong_matches = [s for s in similarities if s > 0.75]

        consistency = len(strong_matches) / len(similarities)

        confidence = round(
            ((0.6 * avg_similarity) +
             (0.4 * consistency)) * 100,
            1
        )

        return confidence

    # -----------------------------------
    # Main routing logic
    # -----------------------------------
    def route(self, question):

        category = self.analyzer.analyze(question)

        print(f"\n[Router] Detected category: {category}\n")

        # -----------------------------
        # Document questions (RAG)
        # -----------------------------
        if category == "document_lookup":

            response = self.rag.ask(question)

            distances = response.get("distances")

            confidence = self.calculate_confidence(distances)

            return {
                "answer": response["answer"],
                "confidence": confidence
            }

        # -----------------------------
        # General finance knowledge
        # -----------------------------
        elif category == "general_finance":

            answer = self.llm.generate(question)

            return {
                "answer": answer,
                "confidence": 65.0
            }

        # -----------------------------
        # Analysis questions (RAG + LLM)
        # -----------------------------
        elif category == "analysis":

            rag_response = self.rag.ask(question)

            distances = rag_response.get("distances")

            confidence = self.calculate_confidence(distances)

            context = rag_response["answer"]

            answer = self.llm.generate(question, context)

            return {
                "answer": answer,
                "confidence": confidence
            }

        # -----------------------------
        # Market data questions
        # -----------------------------
        elif category == "market_data":

            asset = self.asset_extractor.extract(question)

            symbol = asset.get("possible_symbol")

            if not symbol:

                return {
                    "answer": "Could not determine the asset symbol.",
                    "confidence": None
                }

            results = self.market_fetcher.fetch_prices(symbol)

            if not results:

                return {
                    "answer": "Could not fetch market data.",
                    "confidence": "0%"
                }

            price, confidence = self.market_verifier.verify(results)

            return {
                "answer": f"{symbol} current price is ${price}",
                "confidence": confidence
            }

        # -----------------------------
        # Forecast questions
        # -----------------------------
        elif category == "forecast":

            if self.forecast_model:

                prediction, prob = self.forecast_model.predict(question)

                confidence = round(prob * 100, 1)

                return {
                    "answer": prediction,
                    "confidence": confidence
                }

            else:

                return {
                    "answer": "Forecast model is not available yet.",
                    "confidence": None
                }

        # -----------------------------
        # Fallback
        # -----------------------------
        else:

            answer = self.llm.generate(question)

            return {
                "answer": answer,
                "confidence": 60.0
            }