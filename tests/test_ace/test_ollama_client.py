import json
import pytest
from unittest.mock import patch, MagicMock
from continual_learning.ace.ollama_client import OllamaClient


class TestOllamaClient:
    def setup_method(self):
        self.client = OllamaClient(base_url="http://localhost:11434", model="qwen2.5:7b")

    def test_init_stores_config(self):
        assert self.client.model == "qwen2.5:7b"
        assert self.client.base_url == "http://localhost:11434"

    def test_generate_calls_ollama_api(self):
        fake_response = json.dumps({"response": "Hello world"}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = fake_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            result = self.client.generate("Say hello")
            assert result == "Hello world"
            mock_urlopen.assert_called_once()
            call_args = mock_urlopen.call_args
            req = call_args[0][0]
            body = json.loads(req.data)
            assert body["model"] == "qwen2.5:7b"
            assert body["prompt"] == "Say hello"
            assert body["stream"] is False

    def test_generate_with_system_prompt(self):
        fake_response = json.dumps({"response": "I am a reflector"}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = fake_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = self.client.generate("Reflect on this", system="You are a reflector")
            assert result == "I am a reflector"

    def test_generate_handles_connection_error(self):
        with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
            with pytest.raises(ConnectionError):
                self.client.generate("Hello")
