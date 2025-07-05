import os
from langchain_community.document_loaders import PyMuPDFLoader

def load_pdf(path: str):
    loader = PyMuPDFLoader(path)
    docs = loader.load()
    filename = os.path.basename(path)
    for doc in docs:
        doc.metadata["source"] = filename
        # Ensure we always store page_number
        if "page_number" not in doc.metadata:
            doc.metadata["page_number"] = doc.metadata.get("page", "?")
    return docs

