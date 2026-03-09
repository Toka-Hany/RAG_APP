import ollama


class LLMEngine:

    def __init__(self, model="iKhalid/ALLaM:7b"):
        self.model = model

    def generate(self, question, context):

        prompt = f"""
You are a financial assistant.

Use ONLY the provided context to answer the question.

IMPORTANT RULES:
- Do NOT modify numbers.
- Do NOT round numbers.
- Do NOT convert units (thousand → million etc.).
- Copy numbers exactly as written in the context.

Context:
{context}

Question:
{question}

If the answer is not in the context, say:
"The information is not available in the provided documents."
"""



        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response["message"]["content"]