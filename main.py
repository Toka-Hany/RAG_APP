from rag.ingestion.ingestion_engine import ingest_folder
from rag.rag_engine import RAGEngine

data_folder = "data"

# ingest documents first
ingest_folder(data_folder)

rag = RAGEngine()

while True:

    question = input("\nAsk a question (type exit to stop): ")

    if question.lower() == "exit":
        break

    answer = rag.ask(question)

    print("\nAnswer:\n")
    print(answer)