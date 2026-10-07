from langchain_ollama import ChatOllama


LLM_MODEL = "llama3.2:3b"


def get_llm():
    """
    Create the local Ollama language model.
    """

    return ChatOllama(
        model=LLM_MODEL,
        temperature=0,
    )


def build_context(documents):
    """
    Convert retrieved policy documents into a context string.
    """

    context_parts = []

    for document in documents:
        source = document.metadata.get("source", "Unknown document")
        page = document.metadata.get("page", "Unknown")

        context_parts.append(
            f"""
SOURCE: {source}
PAGE: {page}

{document.page_content}
"""
        )

    return "\n\n---\n\n".join(context_parts)


def ask_policy_question(question, documents):
    """
    Answer a question using only the retrieved policy context.
    """

    if not documents:
        return (
            "I couldn't find relevant information in the uploaded "
            "company policies."
        )

    context = build_context(documents)

    prompt = f"""
You are PolicyMind AI, a company policy assistant.

Answer the employee's question using ONLY the policy information
provided in the context below.

Important rules:

1. Do not use outside knowledge.
2. Do not invent policy rules.
3. If the answer is not supported by the context, clearly say:
   "I couldn't find this information in the uploaded policies."
4. Give a concise and professional answer.
5. When possible, mention the relevant document and page.
6. If the policy contains a specific number, date, limit, or
   requirement, preserve it accurately.

POLICY CONTEXT
================

{context}

EMPLOYEE QUESTION
=================

{question}

ANSWER:
"""

    llm = get_llm()

    response = llm.invoke(prompt)

    return response.content