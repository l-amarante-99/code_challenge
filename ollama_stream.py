import httpx
import json

def stream_ollama(model_name, system_prompt, user_prompt):
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
