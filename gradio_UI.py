import gradio as gr
import os
import time
from pdf_loader import load_pdf
from text_splitter import split_documents
from vector_store import build_vector_store
from chatbot import retrieve_answers
from langchain_ollama import OllamaLLM
from ollama_stream import stream_ollama
from concurrent.futures import ThreadPoolExecutor
import hashlib

file_cache = {} 
active_files = set()
vectorstore = None

# Instantiate TinyLlama
llm = OllamaLLM(model="tinyllama")

def hash_file(path):
    """Compute a hash of a PDF file for caching."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def handle_upload(files):
    global vectorstore, file_cache, active_files

    # Determine current filenames in the UI
    current_filenames = set(os.path.basename(path) for path in files)

    # Identify which PDFs were removed
    removed_files = active_files - current_filenames

    # Purge vectorstore docs from removed PDFs
    if vectorstore is not None and removed_files:
        remaining_docs = []
        for doc in vectorstore.docstore._dict.values():
            source = doc.metadata.get("source", None)
            if source and source not in removed_files:
                remaining_docs.append(doc)
        vectorstore = build_vector_store(remaining_docs) if remaining_docs else None

    # Identify new files not loaded yet
    new_files = [
        path for path in files
        if os.path.basename(path) not in file_cache
    ]

    # Load new files in parallel
    loaded_documents = []
    if new_files:
        with ThreadPoolExecutor() as executor:
            loaded_lists = list(executor.map(load_pdf, new_files))
        for docs in loaded_lists:
            loaded_documents.extend(docs)
            if docs:
                filename = docs[0].metadata.get("source")
                if filename:
                    file_cache[filename] = docs

    # Also include any cached docs
    for filename in current_filenames:
        if filename in file_cache:
            loaded_documents.extend(file_cache[filename])

    # Split loaded docs
    split_docs = split_documents(loaded_documents, chunk_size=1000, chunk_overlap=100)

    # Update vectorstore
    if split_docs:
        if vectorstore is None:
            vectorstore = build_vector_store(split_docs)
        else:
            vectorstore.add_documents(split_docs)

    # Update active files
    active_files = current_filenames

    return f"✅ Loaded {len(files)} PDFs. Vector store now contains {len(split_docs)} new chunks."

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

        context = full_text[:8000]

        question = (
            "Please summarize the main topics, findings, and conclusions from the "
            "provided PDF documents using the context below. Ignore instructions and disclaimers."
        )

        all_sources = set()
        for res in vectorstore.docstore._dict.values():
            source = res.metadata.get("source", "unknown.pdf")
            all_sources.add(source)

        citation_text = "\n\nSources used:\n"
        for i, file in enumerate(sorted(all_sources), start=1):
            citation_text += f"{i}. {file}, pages: all pages were used for the summary.\n"

    else:
        # Normal retrieval for non-summary queries
        results = vectorstore.similarity_search_with_score(
            question,
            k=5
        )

        filtered_results = []
        for doc, score in results:
            if score is not None and score <= 0.7:
                filtered_results.append(doc)

        if not filtered_results:
            yield "<div class='gr-box'>No matching content found for this question.</div>"
            return


        file_pages = {}
        context = ""

        for res in filtered_results:
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
