from rag.routing.router import Router

router = Router()

print("Financial AI Assistant Ready.")
print("Type 'exit' to stop.\n")

while True:

    question = input("Ask a question: ")

    if question.lower() == "exit":
        break

    response = router.route(question)

    print("\nAnswer:\n")
    print(response.get("answer", "No answer returned."))

    # Print source if available
    if "source" in response:
        print("\nSource:", response["source"])

    # Print confidence if available
    if "confidence" in response:
        print("\nConfidence:", response["confidence"])

    # Print detected numbers if available
    if "numbers" in response:
        print("\nNumbers detected:")
        print(response["numbers"])

    print("\n" + "-" * 50 + "\n")