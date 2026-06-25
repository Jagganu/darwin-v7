"""
DARWIN v7 — Unified LLM Client

Supports:
  - Ollama          (local, free)
  - Claude API      (Anthropic)
  - NVIDIA NIM      (Kimi K2.6, Llama, etc. — free preview)
  - OpenAI-compatible (OpenRouter, Together, etc.)

Priority order (configurable in config.py):
  1. Try PRIMARY_PROVIDER
  2. On failure, fall back to FALLBACK_PROVIDER
"""

import json
import os
import requests
from typing import Optional
from rich.console import Console

console = Console()


class LLMClient:
    def __init__(
        self,
        primary_provider: str,
        primary_model: str,
        fallback_provider: str = "",
        fallback_model: str = "",
        # Provider configs
        ollama_url: str = "http://localhost:11434",
        anthropic_api_key: str = "",
        nvidia_api_key: str = "",
        openai_compatible_url: str = "",
        openai_compatible_key: str = "",
    ):
        self.primary_provider    = primary_provider
        self.primary_model       = primary_model
        self.fallback_provider   = fallback_provider
        self.fallback_model      = fallback_model

        self.ollama_url          = ollama_url
        self.anthropic_api_key   = anthropic_api_key
        self.nvidia_api_key      = nvidia_api_key
        self.openai_url          = openai_compatible_url
        self.openai_key          = openai_compatible_key

        console.print(
            f"[dim]LLM: primary={primary_provider}/{primary_model} "
            f"fallback={fallback_provider or 'none'}[/dim]"
        )

    # ── Provider Calls ─────────────────────────────────────────────────────

    def _call_ollama(self, prompt: str, model: str, temperature: float,
                     json_mode: bool) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"

        resp = requests.post(
            f"{self.ollama_url}/api/generate",
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["response"]

    def _call_anthropic(self, prompt: str, model: str, temperature: float,
                        json_mode: bool) -> str:
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        system = "Respond only in valid JSON. No explanation, no markdown fences." \
                 if json_mode else "You are a helpful GPU/CPU architecture research AI."

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 2048,
                "temperature": temperature,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]

    def _call_openai_compatible(self, prompt: str, model: str, temperature: float,
                                json_mode: bool, base_url: str, api_key: str) -> str:
        """
        Works for:
          - NVIDIA NIM: base_url = https://integrate.api.nvidia.com/v1
          - OpenRouter:  base_url = https://openrouter.ai/api/v1
          - Together AI: base_url = https://api.together.xyz/v1
        """
        if not api_key:
            raise ValueError(f"API key not set for {base_url}")

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 2048,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    # ── Unified Call ───────────────────────────────────────────────────────

    def _call_provider(self, provider: str, model: str, prompt: str,
                       temperature: float, json_mode: bool) -> str:
        if provider == "ollama":
            return self._call_ollama(prompt, model, temperature, json_mode)

        elif provider == "anthropic":
            return self._call_anthropic(prompt, model, temperature, json_mode)

        elif provider == "nvidia":
            return self._call_openai_compatible(
                prompt, model, temperature, json_mode,
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=self.nvidia_api_key,
            )

        elif provider == "openai_compatible":
            return self._call_openai_compatible(
                prompt, model, temperature, json_mode,
                base_url=self.openai_url,
                api_key=self.openai_key,
            )

        else:
            raise ValueError(f"Unknown provider: {provider}")

    def call(
        self,
        prompt: str,
        temperature: float = 0.5,
        json_mode: bool = True,
    ) -> str:
        """
        Main entry point. Tries primary provider, falls back on failure.
        All modules use this instead of calling requests directly.
        """
        # Try primary
        try:
            result = self._call_provider(
                self.primary_provider, self.primary_model,
                prompt, temperature, json_mode
            )
            return result
        except Exception as e:
            console.print(f"[yellow]Primary ({self.primary_provider}) failed: {e}[/yellow]")

        # Try fallback
        if self.fallback_provider:
            try:
                console.print(f"[cyan]Falling back to {self.fallback_provider}...[/cyan]")
                result = self._call_provider(
                    self.fallback_provider, self.fallback_model,
                    prompt, temperature, json_mode
                )
                return result
            except Exception as e:
                console.print(f"[red]Fallback ({self.fallback_provider}) failed: {e}[/red]")

        raise RuntimeError("All LLM providers failed")

    def call_json(self, prompt: str, temperature: float = 0.5) -> dict:
        """Call LLM and parse response as JSON."""
        raw = self.call(prompt, temperature=temperature, json_mode=True)
        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    def test_connection(self) -> bool:
        """Quick health check."""
        try:
            result = self.call_json(
                'Respond with {"status": "ok"}', temperature=0.0
            )
            ok = result.get("status") == "ok"
            if ok:
                console.print(f"[green]✓ LLM connection OK ({self.primary_provider})[/green]")
            return ok
        except Exception as e:
            console.print(f"[red]✗ LLM connection failed: {e}[/red]")
            return False
