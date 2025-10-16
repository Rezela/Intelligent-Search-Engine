import requests
import json

class HKGAIClient:
    def __init__(self):
        self.base_url = "https://oneapi.hkgai.net/v1"
        self.api_key = "sk-iqA1pjC48rpFXdkU7cCaE3BfBc9145B4BfCbEe0912126646"
        self.model_id = "HKGAI-V1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat(self, system_prompt, user_prompt, max_tokens=500, temperature=0.7):
        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        try:
            response = requests.post(endpoint, headers=self.headers, json=payload)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

        data = response.json()
        content = ""
        try:
            choices = data.get("choices", [])
            if choices:
                first = choices[0] if isinstance(choices[0], dict) else {}
                # Chat schema
                message = first.get("message") or {}
                content = (message.get("content") or "").strip()
                # Fallback to text-based schema
                if not content:
                    content = (first.get("text") or "").strip()
        except Exception:
            pass

        if not content:
            finish_reason = None
            try:
                finish_reason = data.get("choices", [{}])[0].get("finish_reason")
            except Exception:
                pass
            return {
                "content": "",
                "warning": "Empty content returned. Possible causes: wrong endpoint for model, content filter, or max_tokens too small.",
                "finish_reason": finish_reason,
                "raw": data
            }

        return {"content": content, "raw": data}

if __name__ == "__main__":
    client = HKGAIClient()
    system_prompt = "You are a helpful AI assistant providing concise and accurate responses."
    # user_prompt = "what is the capital of China?"
    user_prompt = "What are some common symptoms of hay fever?"
    result = client.chat(system_prompt, user_prompt)
    print(json.dumps(result, indent=2))