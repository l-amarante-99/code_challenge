def retrieve_answers(vectorstore, query, k=3):
    results = vectorstore.similarity_search(query, k=k)
    return results
