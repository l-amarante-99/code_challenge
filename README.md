# PDF Chatbot

An interactive PDF chatbot that lets you upload multiple PDF documents and query their content using a local LLM (TinyLlama via Ollama). Built with Gradio for the UI and LangChain for retrieval and embeddings.

---

## Features

- Upload and parse multiple PDFs
- Semantic search over PDF contents
- Summarize documents
- Streaming responses from a local LLM (TinyLlama)
- Local-only architecture (no cloud APIs required)
- Beautiful Gradio interface with custom styling
- Dynamic updates when PDFs are removed from the upload list

---

## Project Structure

```
project/
│
├── gradio_UI.py               # Main Gradio user interface
├── pdf_loader.py              # Load and parse PDFs into LangChain documents
├── text_splitter.py           # Split documents into chunks for embedding
├── vector_store.py            # Build and manage FAISS vector store
├── chatbot.py                 # Semantic search over vector store
├── ollama_stream.py           # Streaming integration with Ollama LLM
├── README.md                  # Project documentation
```

---

## How It Works

1. Upload PDFs
   - Each PDF is split into smaller text chunks.
   - Chunks are embedded using sentence-transformers (MiniLM).
   - All embeddings are stored in a local FAISS vector database.

2. Ask Questions
   - User queries are converted to embeddings.
   - FAISS retrieves the most relevant chunks.
   - Context is passed to TinyLlama via Ollama’s local API.

3. Get Streaming Answers
   - Responses from TinyLlama stream back into the UI in real-time.
   - The UI shows sources and page numbers used for the answer.

---

## Installation

1. Clone the repository

```bash
git https://github.com/l-amarante-99/code_challenge.git
cd code_challenge
```

2. Create a virtual environment

Recommended: [Conda](https://docs.conda.io/en/latest/):

```bash
conda create -n pdfbot python=3.10
conda activate pdfbot
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies include:

- gradio
- langchain
- sentence-transformers
- langchain_community
- httpx
- PyMuPDF
- faiss-cpu

4. Install and run Ollama

See [Ollama installation guide](https://ollama.com/).

Start Ollama:

```bash
ollama serve
```

Pull TinyLlama:

```bash
ollama run tinyllama
```

---

## Run the App

```bash
python gradio_UI.py
```

Visit [http://localhost:7860](http://localhost:7860) in your browser.

---

## Privacy

All processing is entirely local:

- No PDFs are uploaded to external servers.
- LLM runs locally via Ollama.

---

## Example Questions

- “Summarize the text.”
- “What are the key findings in the paper?”
- “What methods did the authors use?”
- “What are the conclusions of the paper?”

---

## License

[Apache 2.0 License](LICENSE)

---

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [Gradio](https://www.gradio.app/)
- [Ollama](https://ollama.com/)
- [sentence-transformers](https://www.sbert.net/)
