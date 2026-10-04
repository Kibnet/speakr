"""Language models without an API key (#403).

A local server (Ollama, vLLM, llama.cpp) often runs without a key. Every
model call refused to run when TEXT_MODEL_API_KEY was empty, although the
client was built with a placeholder for exactly this case, and the chat
client was never created at all. A dedicated chat model was only used when
it had its own key.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.services.llm as llm


class _FakeOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(llm, "OpenAI", _FakeOpenAI)
    main = object()
    monkeypatch.setattr(llm, "client", main)

    def configure(text_key=None, text_url="http://llm.lan/v1", chat_key=None, chat_url=None, chat_name=None):
        monkeypatch.setattr(llm, "TEXT_MODEL_API_KEY", text_key)
        monkeypatch.setattr(llm, "TEXT_MODEL_BASE_URL", text_url)
        monkeypatch.setattr(llm, "TEXT_MODEL_NAME", "text-model")
        monkeypatch.setattr(llm, "CHAT_MODEL_API_KEY", chat_key)
        monkeypatch.setattr(llm, "CHAT_MODEL_BASE_URL", chat_url)
        monkeypatch.setattr(llm, "CHAT_MODEL_NAME", chat_name)
        return main
    return configure


def test_without_any_key_chat_uses_the_main_client(env):
    main = env()
    assert llm._build_chat_client() is main
    assert llm.get_chat_config()["model_name"] == "text-model"


def test_a_keyless_chat_server_gets_its_own_client(env):
    env(chat_url="http://chat.lan/v1", chat_name="chat-model")
    chat = llm._build_chat_client()
    assert isinstance(chat, _FakeOpenAI)
    assert chat.kwargs["base_url"] == "http://chat.lan/v1" and chat.kwargs["api_key"] == "not-needed"
    assert llm.get_chat_config()["model_name"] == "chat-model"


def test_the_text_key_never_goes_to_another_server(env):
    env(text_key="sk-text", text_url="https://openrouter.ai/api/v1", chat_url="http://chat.lan/v1", chat_name="chat-model")
    assert llm.get_chat_config()["api_key"] is None
    assert llm._build_chat_client().kwargs["api_key"] == "not-needed"


def test_a_chat_model_on_the_text_server_shares_the_key_and_client(env):
    main = env(text_key="sk-text", chat_key="sk-text", chat_name="chat-model")
    assert llm.get_chat_config()["api_key"] == "sk-text"
    assert llm._build_chat_client() is main


def test_a_chat_model_with_its_own_key(env):
    env(text_key="sk-text", chat_key="sk-chat", chat_name="chat-model")
    chat = llm._build_chat_client()
    assert chat.kwargs["api_key"] == "sk-chat" and chat.kwargs["base_url"] == "http://llm.lan/v1"


def test_a_chat_name_alone_keeps_the_text_model(env):
    env(text_key="sk-text", chat_name="chat-model")
    assert llm.get_chat_config()["model_name"] == "text-model"
