import requests
import httpx


class OllamaLLMClient:
    """
    Local free LLM client using Ollama (STABLE VERSION)
    Supports both synchronous (requests) and asynchronous (httpx) generation.
    """

    _OPTIONS = {
        "temperature": 0.3,
        "num_ctx": 2048,
        "num_predict": 512,
    }
    _TIMEOUT = 120

    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def _build_payload(self, prompt: str) -> dict:
        return {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": self._OPTIONS,
        }

    def generate(self, prompt: str) -> str:
        response = requests.post(
            self.url,
            json=self._build_payload(prompt),
            timeout=self._TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    async def generate_async(self, prompt: str) -> str:
        """Async variant using httpx — does not block the event loop."""
        async with httpx.AsyncClient(timeout=self._TIMEOUT) as client:
            response = await client.post(
                self.url, json=self._build_payload(prompt)
            )
            response.raise_for_status()
            return response.json()["response"].strip()
