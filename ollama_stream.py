import httpx
import json

def stream_ollama(model_name, system_prompt, user_prompt):
    """
    Streams text generation output from a locally running Ollama model 
    via its HTTP chat API. Sends a user prompt and receives incremental 
    partial completions as they arrive.

    Args:
        model_name (str):
            The name of the local Ollama model to use (e.g. "tinyllama").
        system_prompt (str):
            A prompt that sets the system's instructions or role.
        user_prompt (str):
            The actual user question or message to be processed.

    Yields:
        str:
            The progressively accumulating text response from the model
            after each streamed chunk. Each yielded value includes all 
            text generated so far.
    """
    url = "http://localhost:11434/api/chat"
    headers = {"Content-Type": "application/json"}

    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": True
    }

    timeout = httpx.Timeout(
        connect=5.0,
        read=None,
        write=None,
        pool=None
    )

    with httpx.stream(
        "POST",
        url,
        headers=headers,
        json=data,
        timeout=timeout
    ) as response:
        accumulated = ""
        for chunk in response.iter_text():
            if chunk.strip() == "":
                continue
            try:
                data = json.loads(chunk)
                delta = data.get("message", {}).get("content", "")
                accumulated += delta
                yield accumulated
            except json.JSONDecodeError:
                continue
