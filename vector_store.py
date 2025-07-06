from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_vector_store(documents):
    """
    Creates a FAISS vector store from a list of LangChain Document objects
    using sentence-transformer embeddings.

    This enables efficient semantic similarity search over document chunks.

    Args:
        documents (list[Document]):
            The list of LangChain Document objects to embed and store.

    Returns:
        FAISS:
            A LangChain FAISS vector store object containing the embeddings
            and allowing similarity search.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

