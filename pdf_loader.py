import os
from langchain_community.document_loaders import PyMuPDFLoader

def load_pdf(path: str):
    """
    Loads a PDF file and converts it into a list of LangChain Document
    objects, each representing a page or chunk of the PDF.

    This function ensures:
    - The document's 'source' metadata is set to the filename.
    - Every document has a 'page_number' field in its metadata for
      easier tracking and referencing.

    Args:
        path (str):
            The full path to the PDF file.

    Returns:
        list[Document]:
            A list of LangChain Document objects extracted from the PDF.
    """
    loader = PyMuPDFLoader(path)
    docs = loader.load()
    filename = os.path.basename(path)
    for doc in docs:
        doc.metadata["source"] = filename
        # Ensure we always store page_number
        if "page_number" not in doc.metadata:
            doc.metadata["page_number"] = doc.metadata.get("page", "?")
    return docs

