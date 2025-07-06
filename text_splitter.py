from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    """
    Splits a list of LangChain Document objects into smaller overlapping
    text chunks for embedding and retrieval.

    Uses LangChain's RecursiveCharacterTextSplitter, which:
    - preserves context by overlapping tokens
    - avoids breaking apart words mid-chunk
    - splits intelligently on characters, sentences, or paragraphs

    Args:
        documents (list[Document]):
            A list of LangChain Document objects to split.
        chunk_size (int, optional):
            The maximum number of characters in each chunk. Default is 1000.
        chunk_overlap (int, optional):
            The number of characters that overlap between consecutive chunks.
            Default is 100.

    Returns:
        list[Document]:
            A new list of LangChain Document objects, each containing a
            text chunk and inherited metadata from the original documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)
