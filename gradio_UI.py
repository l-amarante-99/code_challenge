import gradio as gr
import os
import time
from pdf_loader import load_pdf
from text_splitter import split_documents
from vector_store import build_vector_store
from chatbot import retrieve_answers
from langchain_ollama import OllamaLLM
from ollama_stream import stream_ollama


# Keep a global vectorstore for the current session
vectorstore = None

# Instantiate TinyLlama
llm = OllamaLLM(model="tinyllama")

def handle_upload(files):
    global vectorstore

    all_documents = []
    for file_path in files:
        documents = load_pdf(file_path)
        all_documents.extend(documents)

    split_docs = split_documents(all_documents, chunk_size=1000, chunk_overlap=100)

    if vectorstore is None:
        vectorstore = build_vector_store(split_docs)
    else:
        vectorstore.add_documents(split_docs)

    return f"✅ Loaded {len(files)} PDFs. Vector store now contains {len(split_docs)} new chunks."

from ollama_stream import stream_ollama

def answer_question(files, question):
    global vectorstore

    if vectorstore is None:
        yield "<div class='gr-box'>⚠️ Please upload PDFs first.</div>"
        return

    # Check if the question is a summary request
    is_summary = question.strip().lower() in ["summarize", "summarize the text"]

    if is_summary:
        # Skip retrieval — build full text from all docs
        full_text = ""
        for res in vectorstore.docstore._dict.values():
            full_text += res.page_content.strip() + "\n"

        # Truncate to safe size for small models like TinyLlama
        context = full_text[:8000]

        # Rewrite the question for clarity
        question = (
            "Please summarize the main topics, findings, and conclusions from the "
            "provided PDF documents using the context below. Ignore instructions and disclaimers."
        )

        # Build citations listing all PDFs
        all_sources = set()
        for res in vectorstore.docstore._dict.values():
            source = res.metadata.get("source", "unknown.pdf")
            all_sources.add(source)

        citation_text = "\n\nSources used:\n"
        for i, file in enumerate(sorted(all_sources), start=1):
            citation_text += f"{i}. {file}, pages: all pages were used for the summary.\n"

    else:
        # Normal retrieval for non-summary queries
        results = retrieve_answers(vectorstore, question, k=3)

        if not results:
            yield "<div class='gr-box'>No matching content found.</div>"
            return

        file_pages = {}
        context = ""

        for res in results:
            page = res.metadata.get("page_number", "?")
            if isinstance(page, (list, tuple)):
                page = page[0]
            source = res.metadata.get("source", "unknown.pdf")
            chunk_text = res.page_content.strip().replace("\n", " ")

            context += f"\n[Page {page} — {source}]\n{chunk_text}\n"

            if source not in file_pages:
                file_pages[source] = set()
            if page != "?":
                file_pages[source].add(page)

        # Build citation text for normal queries
        citation_text = "\n\nSources used:\n"
        for i, (file, pages) in enumerate(sorted(file_pages.items()), start=1):
            if pages:
                sorted_pages = sorted(
                    pages, key=lambda x: int(x) if str(x).isdigit() else x
                )
                pages_str = ", ".join(str(p) for p in sorted_pages)
                citation_text += f"{i}. {file}, pages: {pages_str}\n"
            else:
                citation_text += f"{i}. {file}\n"

    # System prompt remains simple
    system_prompt = """
    You are an assistant helping to answer questions about uploaded PDF documents.
    Answer concisely using the provided context.
    """
    
    prompt = f"""Use the following context to answer the question below. 
If the answer isn't contained in the context, say "I couldn't find that information."

Context:
{context}

Question:
{question}
"""

    for partial_text in stream_ollama("tinyllama", system_prompt, prompt):
        combined_text = f"{partial_text}{citation_text}"
        html_output = f"""
        <div class="gr-box" style="display: block; padding: 16px; margin-top: 16px;">
            <div id="output">{combined_text}</div>
        </div>
        """
        yield html_output

# Define custom CSS
css = """
body, .gradio-container {
    background-color: #e6f0fa;
}

button {
    background: linear-gradient(90deg, #70A1D7, #8A89C0) !important;
    color: white !important;
    border: none !important;
}

.gr-box {
    border: 2px solid #70A1D7 !important;
    border-radius: 8px;
    background-color: white;
}

#output {
    min-height: 20px;
    overflow-y: auto;
    color: #333333;
    font-size: 16px;
    white-space: pre-wrap;
}
"""

theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")

with gr.Blocks(theme=theme, css=css) as demo:
    gr.Markdown("## PDF Chatbot")
    gr.Markdown(
        "Upload PDFs and ask questions. "
        "The chatbot will search your documents and generate answers using TinyLlama."
    )

    upload_ui = gr.File(
        file_count="multiple",
        type="filepath",
        label="Upload your PDFs"
    )

    question_ui = gr.Textbox(
        label="Ask a question about your PDFs",
        placeholder="e.g. What are the conclusions of the paper?",
        lines=2
    )

    with gr.Group():
        output_ui = gr.HTML(elem_id="output")

    upload_ui.change(
        fn=handle_upload,
        inputs=[upload_ui],
        outputs=output_ui
    )

    with gr.Row():
        clear_btn = gr.Button("Clear")
        submit_btn = gr.Button("Submit", variant="primary")

    submit_btn.click(
        fn=answer_question,
        inputs=[upload_ui, question_ui],
        outputs=output_ui
    )

    clear_btn.click(
        lambda: "",
        None,
        output_ui
    )

demo.launch()
