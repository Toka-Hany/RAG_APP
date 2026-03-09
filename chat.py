from rag.rag_engine import RAGEngine

rag = RAGEngine()

print("Financial RAG Assistant Ready.")
print("Type 'exit' to stop.\n")

while True:

    question = input("Ask a question: ")

    if question.lower() == "exit":
        break

    response = rag.ask(question)

    print("\nAnswer:\n")
    print(response["answer"])

    print("\nNumbers detected:")
    print(response["numbers_found"])

    print("\n" + "-" * 50 + "\n")