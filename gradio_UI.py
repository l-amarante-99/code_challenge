import gradio as gr
import os
import time
from pdf_loader import load_pdf
from text_splitter import split_documents
from vector_store import build_vector_store
from chatbot import retrieve_answers
from langchain_ollama import OllamaLLM

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

def answer_question(question, progress=gr.Progress(track_tqdm=True)):
    global vectorstore

    if vectorstore is None:
        yield "<div class='output-box'>⚠️ Please upload PDFs first.</div>"
        return

    for i in range(50):
        time.sleep(0.02)
        progress((i + 1) / 50)

    results = retrieve_answers(vectorstore, question, k=10)

    if not results:
        yield "<div class='output-box'>No matching content found.</div>"
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

    prompt = f"""You are an assistant helping to answer questions about uploaded PDF documents.

When asked "What is the text about?", produce a summary of the overall topics and contributions of the documents, in plain language. 
Do not simply list references or citations unless specifically asked.
When asked "What are the conclusions of the paper?", provide a concise summary of the main conclusions drawn in the documents.
When asked "What are the key findings?", summarize the most important findings or results presented in the documents.

Use the following context to answer the question below. 
If the answer isn't contained in the context, say "I couldn't find that information."

Context:
{context}

Question:
{question}
"""

    answer = llm.invoke(prompt)

    # Format grouped citations
    if file_pages:
        citation_text = "\n\nSources used:\n"
        for i, (file, pages) in enumerate(sorted(file_pages.items()), start=1):
            if pages:
                sorted_pages = sorted(pages, key=lambda x: int(x) if str(x).isdigit() else x)
                pages_str = ", ".join(str(p) for p in sorted_pages)
                citation_text += f"{i}. {file}, pages: {pages_str}\n"
            else:
                citation_text += f"{i}. {file}\n"
    else:
        citation_text = "\n\nNo sources found."

    html_output = f"<div class='output-box'>{answer}{citation_text}</div>"

    yield html_output

# Define your custom CSS
css = """
body, .gradio-container {
    background-color: #e6f0fa;
}

.gr-button-primary {
    background: linear-gradient(90deg, #70A1D7, #8A89C0);
    color: white;
    border: none;
}

.output-box {
    background-color: white;
    border: 2px solid #70A1D7;
    border-radius: 8px;
    padding: 16px;
    margin-top: 16px;
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

    output_ui = gr.HTML()

    with gr.Row():
        clear_btn = gr.Button("Clear")
        submit_btn = gr.Button("Submit", variant="primary")

    def full_pipeline(files, question):
        upload_msg = handle_upload(files)
        for result in answer_question(question):
            pass
        return result

    submit_btn.click(
        fn=full_pipeline,
        inputs=[upload_ui, question_ui],
        outputs=output_ui
    )

    clear_btn.click(
        lambda: "",
        None,
        output_ui
    )

demo.launch()
