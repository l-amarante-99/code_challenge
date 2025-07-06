def retrieve_answers(vectorstore, query, k=3):
    """
    Retrieves the top-k most semantically similar document chunks
    from a vector store given a user query.

    This function performs semantic search using vector similarity.

    Args:
        vectorstore (FAISS or BaseVectorStore):
            The vector store object containing document embeddings.
        query (str):
            The user question or search query.
        k (int, optional):
            The number of top matching documents to retrieve.
            Default is 3.

    Returns:
        list[Document]:
            A list of LangChain Document objects ranked by similarity
            to the query.
    """
    results = vectorstore.similarity_search(query, k=k)
    return results
