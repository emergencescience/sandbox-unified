import os

# Set environment variables before importing the app
os.environ["GEMINI_API_KEY"] = "test_gemini_key"
os.environ["OPENAI_API_KEY"] = "test_openai_key"
os.environ["DEFAULT_VLM_PROVIDER"] = "gemini"

from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app, VerifyResponse

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "OK"

def test_verify_no_image():
    # Manual validation handles this now as both base64 and url are optional in the pydantic model
    response = client.post("/verify/llm", json={"prompt": "test"})
    assert response.status_code == 400
    assert "Image data is required" in response.json()["detail"]

@patch("main.call_gemini")
def test_verify_gemini_success(mock_gemini):
    mock_gemini.return_value = VerifyResponse(
        pass_status=True,
        reason="Looks good",
        confidence=0.95,
        raw_output='{"pass": true, "reason": "Looks good", "confidence": 0.95}'
    )
    
    payload = {
        "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
        "prompt": "Is this a white pixel?",
        "provider": "gemini"
    }
    response = client.post("/verify/llm", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["pass_status"] is True
    assert data["confidence"] == 0.95
    mock_gemini.assert_called_once()

@patch("main.call_openai")
def test_verify_openai_success(mock_openai):
    mock_openai.return_value = VerifyResponse(
        pass_status=False,
        reason="Not a circle",
        confidence=0.8,
        raw_output='{"pass": false, "reason": "Not a circle", "confidence": 0.8}'
    )
    
    payload = {
        "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
        "prompt": "Is this a circle?",
        "provider": "openai"
    }
    response = client.post("/verify/llm", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["pass_status"] is False
    assert data["confidence"] == 0.8
    mock_openai.assert_called_once()
