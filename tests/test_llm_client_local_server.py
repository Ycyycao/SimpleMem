"""What the client sends a local OpenAI-compatible server: nothing extra unless asked for."""
from types import SimpleNamespace

import pytest

from simplemem.core.utils.llm_client import LLMClient


def sent(monkeypatch, **env):
    for k in ("SIMPLEMEM_TEMPLATE_THINKING", "SIMPLEMEM_MAX_TOKENS"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    client = LLMClient(api_key="k", model="m", base_url="http://local/v1", enable_thinking=False, use_streaming=False)
    calls = []

    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    monkeypatch.setattr(client.client.chat.completions, "create", create)
    assert client.chat_completion([{"role": "user", "content": "x"}]) == "ok"
    return calls[0]


def test_unset_sends_neither(monkeypatch):
    kw = sent(monkeypatch)
    assert "extra_body" not in kw and "max_tokens" not in kw


def test_thinking_setting_reaches_the_chat_template(monkeypatch):
    kw = sent(monkeypatch, SIMPLEMEM_TEMPLATE_THINKING="1")
    assert kw["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_output_cap(monkeypatch):
    assert sent(monkeypatch, SIMPLEMEM_MAX_TOKENS="16384")["max_tokens"] == 16384
