"""
Multi-API Provider Layer for CineSense AI – Phase 3.
Supports: Gemma 3n + Gemma 27B (raw requests) + Phi-4 (OpenAI client).
All APIs route through https://integrate.api.nvidia.com/v1.
"""
import os
import json
import requests
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()


class BaseProvider(ABC):
    """Abstract base class for all LLM API providers."""
    
    @abstractmethod
    def get_completion(self, prompt: str, system_prompt: str = "") -> str:
        ...
    
    @abstractmethod
    def get_name(self) -> str:
        ...
    
    @abstractmethod
    def is_available(self) -> bool:
        ...


class GemmaProvider(BaseProvider):
    """
    Google Gemma via raw HTTP requests to NVIDIA's API.
    Uses streaming SSE and collects the full response.
    """
    
    def __init__(self):
        self.api_key = os.getenv("GEMMA_API_KEY")
        self.model = os.getenv("GEMMA_MODEL", "google/gemma-3n-e4b-it")
        self.base_url = os.getenv("NVIDIA_API_BASE_URL", "https://integrate.api.nvidia.com/v1")
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def get_name(self) -> str:
        model_short = self.model.split("/")[-1] if self.model else "gemma"
        return f"Gemma ({model_short})"
    
    def get_completion(self, prompt: str, system_prompt: str = "") -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.20,
            "top_p": 0.70,
            "stream": True,
        }
        
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=10)
        response.raise_for_status()
        
        # Collect streamed SSE chunks into a single response
        full_text = []
        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if decoded.startswith("data: "):
                data_str = decoded[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        full_text.append(content)
                except json.JSONDecodeError:
                    continue
        
        return "".join(full_text)


class Phi4Provider(BaseProvider):
    """
    Microsoft Phi-4-mini-instruct via OpenAI-compatible client → NVIDIA endpoint.
    """
    
    def __init__(self):
        self.api_key = os.getenv("PHI4_API_KEY")
        self.model = os.getenv("PHI4_MODEL", "microsoft/phi-4-mini-instruct")
        self.base_url = os.getenv("NVIDIA_API_BASE_URL", "https://integrate.api.nvidia.com/v1")
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def get_name(self) -> str:
        return f"Phi-4 ({self.model.split('/')[-1]})"
    
    def get_completion(self, prompt: str, system_prompt: str = "") -> str:
        from openai import OpenAI
        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=10.0)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        completion = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            top_p=0.7,
            max_tokens=1024,
            stream=True,
        )
        
        # Collect streamed chunks
        full_text = []
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                full_text.append(chunk.choices[0].delta.content)
        
        return "".join(full_text)


class Gemma27BProvider(BaseProvider):
    """
    Google Gemma-3-27B-IT via raw HTTP requests to NVIDIA's API.
    Larger, more capable model for complex reasoning.
    """
    
    def __init__(self):
        self.api_key = os.getenv("GEMMA27B_API_KEY")
        self.model = os.getenv("GEMMA27B_MODEL", "google/gemma-3-27b-it")
        self.base_url = os.getenv("NVIDIA_API_BASE_URL", "https://integrate.api.nvidia.com/v1")
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def get_name(self) -> str:
        model_short = self.model.split("/")[-1] if self.model else "gemma-27b"
        return f"Gemma 27B ({model_short})"
    
    def get_completion(self, prompt: str, system_prompt: str = "") -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.20,
            "top_p": 0.70,
            "stream": True,
        }
        
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=10)
        response.raise_for_status()
        
        full_text = []
        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if decoded.startswith("data: "):
                data_str = decoded[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        full_text.append(content)
                except json.JSONDecodeError:
                    continue
        
        return "".join(full_text)


class APIProviderManager:
    """
    Factory that manages multiple LLM providers with automatic fallback.
    
    Fallback chain: Preferred API -> next available -> None (local models).
    Default priority: Gemma 3n -> Gemma 27B -> Phi-4.
    """
    
    PROVIDER_MAP = {
        "gemma": GemmaProvider,
        "gemma27b": Gemma27BProvider,
        "phi4": Phi4Provider,
    }
    
    # Default priority order
    DEFAULT_PRIORITY = ["gemma", "gemma27b", "phi4"]
    
    def __init__(self):
        self._providers: dict[str, BaseProvider] = {}
        for key, cls in self.PROVIDER_MAP.items():
            self._providers[key] = cls()
    
    def list_available(self) -> list[str]:
        """Return list of provider keys that have valid API keys configured."""
        return [key for key, provider in self._providers.items() if provider.is_available()]
    
    def get_provider(self, preferred: str = "auto") -> BaseProvider | None:
        """
        Get the best available provider.
        
        Args:
            preferred: "auto", "gemma", "phi4", or "local"
        """
        if preferred == "local":
            return None
        
        if preferred != "auto" and preferred in self._providers:
            provider = self._providers[preferred]
            if provider.is_available():
                return provider
        
        # Auto mode: try each in priority order
        for key in self.DEFAULT_PRIORITY:
            provider = self._providers.get(key)
            if provider and provider.is_available():
                return provider
        
        return None
    
    def get_completion(self, prompt: str, preferred: str = "auto", system_prompt: str = "") -> tuple[str | None, str]:
        """
        Get a completion from the best available provider.
        
        Returns:
            Tuple of (completion_text, provider_name).
            completion_text is None if all APIs failed.
        """
        provider = self.get_provider(preferred)
        if provider is None:
            return None, "local"
        
        try:
            result = provider.get_completion(prompt, system_prompt=system_prompt)
            return result, provider.get_name()
        except Exception as e:
            print(f"API Error ({provider.get_name()}): {e}")
            
            # Try fallback to next available provider
            for key in self.DEFAULT_PRIORITY:
                fallback = self._providers.get(key)
                if fallback and fallback.is_available() and fallback is not provider:
                    try:
                        result = fallback.get_completion(prompt, system_prompt=system_prompt)
                        return result, fallback.get_name()
                    except Exception as e2:
                        print(f"Fallback API Error ({fallback.get_name()}): {e2}")
            
            return None, "local"
