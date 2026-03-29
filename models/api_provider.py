"""
Multi-API Provider Layer for CineSense AI.
Supports:
- Gemini 2.5 Flash via Google's Generative Language API
- Gemma 3n + Gemma 27B via NVIDIA's API
- Phi-4 via OpenAI-compatible NVIDIA endpoint
"""
import base64
import json
import mimetypes
import os
from abc import ABC, abstractmethod
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()


class BaseProvider(ABC):
    """Abstract base class for all API providers."""

    @abstractmethod
    def get_completion(self, prompt: str, system_prompt: str = "") -> str:
        ...

    @abstractmethod
    def get_name(self) -> str:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...

    def can_analyze_images(self) -> bool:
        return False

    def analyze_image(self, image_bytes: bytes, mime_type: str, prompt: str, system_prompt: str = "") -> str:
        raise NotImplementedError(f"{self.get_name()} does not support image analysis.")


class GeminiFlashProvider(BaseProvider):
    """
    Gemini 2.5 Flash via the Generative Language API.
    Uses Google's REST generateContent endpoint for both text and image analysis.
    """

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.base_url = os.getenv(
            "GEMINI_API_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta",
        ).rstrip("/")
        self.timeout = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "20"))

    def is_available(self) -> bool:
        return bool(self.api_key)

    def get_name(self) -> str:
        return f"Gemini ({self.model})"

    def can_analyze_images(self) -> bool:
        return True

    def get_completion(self, prompt: str, system_prompt: str = "") -> str:
        full_prompt = self._merge_prompts(system_prompt, prompt)
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": full_prompt,
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 768,
            },
        }
        return self._generate_content(payload)

    def analyze_image(self, image_bytes: bytes, mime_type: str, prompt: str, system_prompt: str = "") -> str:
        full_prompt = self._merge_prompts(system_prompt, prompt)
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64.b64encode(image_bytes).decode("utf-8"),
                            }
                        },
                        {
                            "text": full_prompt,
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 768,
            },
        }
        return self._generate_content(payload)

    def _generate_content(self, payload: dict[str, Any]) -> str:
        url = f"{self.base_url}/models/{self.model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()

        text = self._extract_text(data)
        if text:
            return text

        feedback = data.get("promptFeedback", {})
        raise ValueError(f"Gemini returned no text output. Prompt feedback: {feedback}")

    @staticmethod
    def _merge_prompts(system_prompt: str, prompt: str) -> str:
        if system_prompt:
            return f"{system_prompt}\n\n{prompt}"
        return prompt

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        text_parts: list[str] = []

        for candidate in data.get("candidates", []):
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                text = part.get("text", "").strip()
                if text:
                    text_parts.append(text)

        return "\n".join(text_parts).strip()


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
    Microsoft Phi-4-mini-instruct via OpenAI-compatible client -> NVIDIA endpoint.
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
    Factory that manages multiple API providers with automatic fallback.

    Text fallback chain: Preferred API -> next available -> None (local models).
    Vision priority: Gemini -> None (local vision model).
    """

    PROVIDER_MAP = {
        "gemini": GeminiFlashProvider,
        "gemma": GemmaProvider,
        "gemma27b": Gemma27BProvider,
        "phi4": Phi4Provider,
    }

    DEFAULT_PRIORITY = ["gemma", "gemma27b", "phi4", "gemini"]
    VISION_PRIORITY = ["gemini"]

    def __init__(self):
        self._providers: dict[str, BaseProvider] = {}
        for key, cls in self.PROVIDER_MAP.items():
            self._providers[key] = cls()

    def list_available(self) -> list[str]:
        """Return list of provider keys that have valid API keys configured."""
        return [key for key, provider in self._providers.items() if provider.is_available()]

    def get_provider(self, preferred: str = "auto") -> BaseProvider | None:
        """
        Get the best available text provider.

        Args:
            preferred: "auto", "gemini", "gemma", "gemma27b", "phi4", or "local"
        """
        if preferred == "local":
            return None

        if preferred != "auto" and preferred in self._providers:
            provider = self._providers[preferred]
            if provider.is_available():
                return provider

        for key in self.DEFAULT_PRIORITY:
            provider = self._providers.get(key)
            if provider and provider.is_available():
                return provider

        return None

    def get_vision_provider(self, preferred: str = "auto") -> BaseProvider | None:
        """Return the best provider that can analyze uploaded images."""
        if preferred == "local":
            return None

        if preferred != "auto":
            provider = self._providers.get(preferred)
            if provider and provider.is_available() and provider.can_analyze_images():
                return provider
            return None

        for key in self.VISION_PRIORITY:
            provider = self._providers.get(key)
            if provider and provider.is_available() and provider.can_analyze_images():
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

            for key in self.DEFAULT_PRIORITY:
                fallback = self._providers.get(key)
                if fallback and fallback.is_available() and fallback is not provider:
                    try:
                        result = fallback.get_completion(prompt, system_prompt=system_prompt)
                        return result, fallback.get_name()
                    except Exception as e2:
                        print(f"Fallback API Error ({fallback.get_name()}): {e2}")

            return None, "local"

    def analyze_image(
        self,
        image_input: Any,
        preferred: str = "auto",
        prompt: str = "",
        system_prompt: str = "",
    ) -> tuple[str | None, str]:
        """
        Analyze an uploaded image with a provider that supports multimodal input.

        Returns:
            Tuple of (analysis_text, provider_name).
            analysis_text is None if no vision provider is configured or all vision APIs fail.
        """
        provider = self.get_vision_provider(preferred)
        if provider is None:
            return None, "local"

        try:
            image_bytes, mime_type = self._read_image_input(image_input)
            if not image_bytes:
                return None, provider.get_name()
            result = provider.analyze_image(
                image_bytes=image_bytes,
                mime_type=mime_type,
                prompt=prompt,
                system_prompt=system_prompt,
            )
            return result, provider.get_name()
        except Exception as e:
            print(f"Vision API Error ({provider.get_name()}): {e}")
            return None, "local"

    @staticmethod
    def _read_image_input(image_input: Any) -> tuple[bytes, str]:
        """Extract raw bytes and mime type from uploaded images or local file paths."""
        if isinstance(image_input, (bytes, bytearray)):
            return bytes(image_input), "image/jpeg"

        if isinstance(image_input, str):
            mime_type = mimetypes.guess_type(image_input)[0] or "image/jpeg"
            with open(image_input, "rb") as file_obj:
                return file_obj.read(), mime_type

        file_name = getattr(image_input, "name", "uploaded-image")
        mime_type = getattr(image_input, "type", None) or mimetypes.guess_type(file_name)[0] or "image/jpeg"

        if hasattr(image_input, "getvalue"):
            return image_input.getvalue(), mime_type

        if hasattr(image_input, "read"):
            current_pos = None
            if hasattr(image_input, "tell"):
                try:
                    current_pos = image_input.tell()
                except Exception:
                    current_pos = None

            image_bytes = image_input.read()
            if not image_bytes and hasattr(image_input, "seek"):
                image_input.seek(0)
                image_bytes = image_input.read()

            if hasattr(image_input, "seek"):
                try:
                    image_input.seek(current_pos if current_pos is not None else 0)
                except Exception:
                    pass

            return image_bytes, mime_type

        raise TypeError("Unsupported image input type for vision analysis.")
