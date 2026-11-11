"""One tiny interface over Anthropic, OpenAI and a local Ollama model."""
import os

import requests

import config


def generate(system: str, prompt: str, max_tokens: int = 1000, temperature: float = 0.2) -> str:
    provider = config.LLM_PROVIDER

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    if provider == "ollama":
        resp = requests.post(
            f"{config.OLLAMA_URL}/api/chat",
            json={
                "model": config.OLLAMA_MODEL,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    raise ValueError("LLM_PROVIDER must be one of: anthropic, openai, ollama")
