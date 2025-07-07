import json
import re
from ollama_stream import stream_ollama

def extract_keywords_with_llm(question: str) -> list[str]:
    """
    Uses TinyLlama to extract the best keywords for retrieval from the user's question.

    Args:
        question (str): The user's question.

    Returns:
        list[str]: A list of extracted keywords.
    """
    system_prompt = """
    You are an expert assistant for document retrieval from PDF texts.

    Your task:
    - Extract only domain-specific technical terms, key concepts, or named entities from the question.
    - Avoid generic words like "model", "approach", "data", "paper", "study", "method", "research", "document", etc.
    - If appropriate, return multi-word keyphrases rather than splitting them into single words.
    - Prefer precise terminology over broad words.

    Example:

    Question:
    "Compare adversarial attacks in vision transformers and how 3D CNNs and saliency mapping mitigate them."

    Output:
    ["adversarial attacks", "vision transformers", "3D CNNs", "saliency mapping"]

    Return only a JSON array of keywords, with no extra explanation.
    """

    user_prompt = f"""
    Question:
    {question}
    """

    # Invoke TinyLlama via your existing streaming helper
    # We'll just read the first chunk since this prompt should be short
    result = ""
    for partial in stream_ollama("tinyllama", system_prompt, user_prompt):
        result += partial

    # Try to parse JSON from the LLM output
    try:
        keywords = json.loads(result.strip())
        return [k for k in keywords if k.strip()]
    except json.JSONDecodeError:
        # fallback: crude regex split
        words = re.findall(r"\w+", question.lower())
        stopwords = set([
            "the", "and", "for", "with", "that", "this", "from", "using", 
            "into", "over", "under", "between", "which", "what", "whose",
            "these", "those", "also", "their", "there", "where", "when",
            "how", "than", "such", "some", "have", "has", "been", "but",
            "can", "could", "will", "would", "should", "may", "might"
        ])
        return [w for w in words if len(w) > 3 and w not in stopwords]
