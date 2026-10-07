from langchain_ollama import OllamaEmbeddings


EMBEDDING_MODEL = "nomic-embed-text"


def get_embeddings():
    """
    Create the local Ollama embedding model.
    """

    return OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )