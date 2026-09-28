import pytest
from yake_sum.backends import get_llm_client, MockLLMClient, BaseLLMClient

def test_mock_backend_generation():
    client = get_llm_client(backend="mock")
    assert isinstance(client, MockLLMClient)
    available, msg = client.is_available()
    assert available is True
    assert "Mock backend" in msg
    output = client.generate("Summarize this: Keywords: [ai, data]\n\nDocument:\nDeep learning uses data.")
    assert "Mock summary" in output
    assert len(output) > 10

def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unsupported backend"):
        get_llm_client(backend="nonexistent_backend")

def test_ollama_client_instantiation():
    client = get_llm_client(backend="ollama", model="mistral", host="http://localhost:11434")
    assert hasattr(client, "generate")
    assert hasattr(client, "is_available")
