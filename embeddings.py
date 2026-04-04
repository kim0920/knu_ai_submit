from langchain_ollama import OllamaEmbeddings

_embedding = OllamaEmbeddings(
        model="nomic-embed-text-v2-moe"
    )

def get_embeddings():
    return _embedding