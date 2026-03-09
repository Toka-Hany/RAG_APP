from rag.retrieval.retriever import Retriever
from rag.llm.llm_engine import LLMEngine


class RAGEngine:

    def __init__(self):

        self.retriever = Retriever()
        self.llm = LLMEngine()

    def ask(self, question):

        docs = self.retriever.search(question)

        context = "\n\n".join(docs)

        answer = self.llm.generate(question, context)

        return answer