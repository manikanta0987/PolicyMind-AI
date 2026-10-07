from langchain_chroma import Chroma


def create_vector_store(documents, embeddings):
    """
    Create an in-memory Chroma vector store from policy chunks.
    """

    if not documents:
        return None

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="policymind_policies",
    )

    return vector_store


def search_policies(vector_store, question, k=4):
    """
    Retrieve the most relevant policy chunks.
    """

    if vector_store is None:
        return []

    results = vector_store.similarity_search(
        question,
        k=k,
    )

    return results
