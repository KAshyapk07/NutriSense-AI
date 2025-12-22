import requests

class OllamaLLMClient:
    """
    Local free LLM client using Ollama (STABLE VERSION)
    """

    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_ctx": 2048,      
                "num_predict": 512    
            }
        }

        response = requests.post(
            self.url,
            json=payload,
            timeout=120            
        )

        response.raise_for_status()

        return response.json()["response"].strip()
