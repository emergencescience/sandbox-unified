import os
import json

# Set environment variables before importing the app
os.environ["GEMINI_API_KEY"] = "test_gemini_key"
os.environ["OPENAI_API_KEY"] = "test_openai_key"
os.environ["OPENROUTER_API_KEY"] = "test_openrouter_key"
os.environ["DEFAULT_VLM_PROVIDER"] = "gemini"

from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app, VerifyResponse

client = TestClient(app)


# ── Health ──────────────────────────────────────────────────────────────

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "OK"
    assert "llm" in response.json()["components"]


# ── Validation ──────────────────────────────────────────────────────────

def test_verify_no_solution_no_image():
    """Must provide either candidate_solution or image_base64."""
    response = client.post("/verify/llm", json={"prompt": "test"})
    assert response.status_code == 400
    assert "candidate_solution" in response.json()["detail"] or "image_base64" in response.json()["detail"]


def test_verify_image_url_only_unsupported():
    """image_url without candidate_solution or image_base64 should return 400."""
    response = client.post("/verify/llm", json={"prompt": "test", "image_url": "https://example.com/img.png"})
    assert response.status_code == 400


# ── VLM (Vision) verification — legacy path ─────────────────────────────

@patch("main.call_gemini")
def test_verify_gemini_vlm_image(mock_gemini):
    """VLM path: image_base64 with gemini provider."""
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
def test_verify_openai_vlm_image(mock_openai):
    """VLM path: image_base64 with openai provider."""
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


def test_verify_gemini_no_api_key():
    """Gemini provider without API key should return 500."""
    import main as main_module
    original_key = main_module.GEMINI_API_KEY
    main_module.GEMINI_API_KEY = None
    try:
        payload = {
            "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
            "prompt": "test",
            "provider": "gemini"
        }
        response = client.post("/verify/llm", json=payload)
        assert response.status_code == 500
        assert "Gemini API key" in response.json()["detail"]
    finally:
        main_module.GEMINI_API_KEY = original_key


def test_verify_unsupported_provider():
    """Provider without slash (not gemini/openai) should return 400."""
    payload = {
        "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
        "prompt": "test",
        "provider": "claude"
    }
    response = client.post("/verify/llm", json=payload)
    assert response.status_code == 400
    assert "not supported" in response.json()["detail"]


# ── LLM (Text) verification — new path via candidate_solution ───────────

@patch("main.call_openrouter")
def test_verify_llm_text_with_openrouter(mock_openrouter):
    """LLM text path: candidate_solution with openrouter provider (deepseek/deepseek-chat)."""
    mock_openrouter.return_value = VerifyResponse(
        pass_status=True,
        reason="Article is well-structured",
        confidence=0.9,
        raw_output='{"pass": true, "reason": "Article is well-structured", "confidence": 0.9}'
    )
    
    payload = {
        "candidate_solution": "# My Article\n\nThis is a well-structured article about AI.",
        "prompt": "Is this article well-structured with proper headings?",
        "provider": "deepseek/deepseek-chat"
    }
    response = client.post("/verify/llm", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["pass_status"] is True
    assert data["confidence"] == 0.9
    mock_openrouter.assert_called_once()
    # Verify text content was passed
    call_args = mock_openrouter.call_args
    assert "# My Article" in call_args[0][1]  # solution_data arg


@patch("main.call_openrouter")
def test_verify_llm_text_rejected(mock_openrouter):
    """LLM text path: candidate_solution gets rejected."""
    mock_openrouter.return_value = VerifyResponse(
        pass_status=False,
        reason="Code has security vulnerability",
        confidence=0.85,
        raw_output='{"pass": false, "reason": "Code has security vulnerability", "confidence": 0.85}'
    )
    
    payload = {
        "candidate_solution": "function unsafe() { eval(userInput); }",
        "prompt": "Is this JavaScript code secure?",
        "provider": "deepseek/deepseek-chat"
    }
    response = client.post("/verify/llm", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["pass_status"] is False
    mock_openrouter.assert_called_once()


@patch("main.call_openrouter")
def test_verify_llm_candidate_solution_priority(mock_openrouter):
    """candidate_solution takes priority over image_base64."""
    mock_openrouter.return_value = VerifyResponse(
        pass_status=True,
        reason="Text verified",
        confidence=0.8,
        raw_output='{"pass": true}'
    )
    
    payload = {
        "candidate_solution": "Some text solution",
        "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
        "prompt": "Evaluate this",
        "provider": "deepseek/deepseek-chat"
    }
    response = client.post("/verify/llm", json=payload)
    
    assert response.status_code == 200
    # Should have called openrouter (text path since provider has "/")
    mock_openrouter.assert_called_once()
    # solution_data should be the candidate_solution text, not image_base64
    call_args = mock_openrouter.call_args
    assert call_args[0][1] == "Some text solution"


@patch("main.call_openrouter")
def test_verify_llm_with_submission_id(mock_openrouter):
    """submission_id should be present in the request and handled."""
    mock_openrouter.return_value = VerifyResponse(
        pass_status=True, reason="OK", confidence=1.0, raw_output='{"pass": true}'
    )
    
    payload = {
        "candidate_solution": "My solution text",
        "prompt": "Is this correct?",
        "provider": "deepseek/deepseek-chat",
        "submission_id": "sub-abc-123"
    }
    response = client.post("/verify/llm", json=payload)
    
    assert response.status_code == 200


@patch("main.call_openrouter")
def test_verify_llm_verify_keyword_as_solution(mock_openrouter):
    """The orchestrator uses 'verify' as the default dry-run solution for LLM bounties.
    This should be treated as a valid candidate_solution text."""
    mock_openrouter.return_value = VerifyResponse(
        pass_status=True, reason="OK", confidence=1.0, raw_output='{"pass": true}'
    )
    
    payload = {
        "candidate_solution": "verify",
        "prompt": "Is this correct?",
        "provider": "deepseek/deepseek-chat"
    }
    response = client.post("/verify/llm", json=payload)
    
    assert response.status_code == 200
    mock_openrouter.assert_called_once()


@patch("main.call_openrouter")
def test_verify_llm_default_provider_uses_openrouter(mock_openrouter):
    """When no provider is specified and DEFAULT_VLM_PROVIDER has a slash, 
    it should route to openrouter."""
    mock_openrouter.return_value = VerifyResponse(
        pass_status=True, reason="OK", confidence=1.0, raw_output='{"pass": true}'
    )
    
    # Save and override default
    import main
    original_default = main.DEFAULT_PROVIDER
    main.DEFAULT_PROVIDER = "deepseek/deepseek-chat"
    
    try:
        payload = {
            "candidate_solution": "Some solution",
            "prompt": "Evaluate this"
        }
        response = client.post("/verify/llm", json=payload)
        assert response.status_code == 200
        mock_openrouter.assert_called_once()
    finally:
        main.DEFAULT_PROVIDER = original_default


# ── call_openrouter unit tests ──────────────────────────────────────────

def test_call_openrouter_text_solution():
    """call_openrouter should send text as a text user message."""
    from main import call_openrouter
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({
        "pass": True, "reason": "Correct", "confidence": 0.95
    })
    
    with patch("main.OpenAI") as mock_openai_cls:
        mock_openai_instance = MagicMock()
        mock_openai_instance.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_openai_instance
        
        result = call_openrouter("Evaluate this text", "Hello world solution", "deepseek/deepseek-chat")
        
        assert result.pass_status is True
        assert result.reason == "Correct"
        
        # Verify the message content is text, not image
        create_call = mock_openai_instance.chat.completions.create.call_args
        messages = create_call.kwargs.get("messages") or create_call[1].get("messages")
        # User message should be a plain string (text), not a list with image_url
        user_msg = messages[1]["content"]
        assert isinstance(user_msg, str), f"Expected string content for text, got {type(user_msg)}"
        assert "Hello world solution" in user_msg


def test_call_openrouter_image_solution():
    """call_openrouter should send base64 image as image_url content."""
    from main import call_openrouter
    
    # Use a base64 image string that's over 100 chars and starts with "iV" 
    # (the auto-detect threshold for base64 images)
    b64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==" + "A" * 100
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({
        "pass": True, "reason": "Image matches", "confidence": 0.9
    })
    
    with patch("main.OpenAI") as mock_openai_cls:
        mock_openai_instance = MagicMock()
        mock_openai_instance.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_openai_instance
        
        result = call_openrouter("Is this a cat?", b64_image, "openai/gpt-4o")
        
        assert result.pass_status is True
        
        # Verify the message content includes image_url
        create_call = mock_openai_instance.chat.completions.create.call_args
        messages = create_call.kwargs.get("messages") or create_call[1].get("messages")
        user_msg = messages[1]["content"]
        assert isinstance(user_msg, list), f"Expected list content for image, got {type(user_msg)}"


def test_call_openrouter_length_limit():
    """call_openrouter should reject text solutions over 5000 chars."""
    from main import call_openrouter
    
    long_text = "A" * 5001
    result = call_openrouter("Evaluate", long_text, "deepseek/deepseek-chat")
    
    assert result.pass_status is False
    assert "length limit" in result.reason.lower()
    assert result.confidence == 1.0


def test_call_openrouter_length_limit_allows_images():
    """call_openrouter should NOT reject long base64 images over 5000 chars."""
    from main import call_openrouter
    
    # A base64 image that's over 5000 chars but starts with "iVBORw0K"
    b64_image = "iVBORw0KGgo" + "A" * 5000
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({
        "pass": True, "reason": "Image OK", "confidence": 0.9
    })
    
    with patch("main.OpenAI") as mock_openai_cls:
        mock_openai_instance = MagicMock()
        mock_openai_instance.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_openai_instance
        
        result = call_openrouter("Evaluate", b64_image, "openai/gpt-4o")
        assert result.pass_status is True


def test_call_openrouter_malformed_json_response():
    """call_openrouter should handle malformed JSON from LLM gracefully."""
    from main import call_openrouter
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "This is not JSON at all"
    
    with patch("main.OpenAI") as mock_openai_cls:
        mock_openai_instance = MagicMock()
        mock_openai_instance.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_openai_instance
        
        result = call_openrouter("Evaluate", "Some text", "deepseek/deepseek-chat")
        assert result.pass_status is False
        assert "Error parsing" in result.reason
        assert result.confidence == 0.0


# ── Code execution tests ────────────────────────────────────────────────

def test_execute_python_success():
    payload = {
        "solution": "def add(a, b): return a + b",
        "test": "from solution import add\nassert add(1, 2) == 3\nprint('OK')",
        "language": "python3"
    }
    response = client.post("/execute/python", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "accepted"


def test_execute_python_failure():
    payload = {
        "solution": "def add(a, b): return a - b",
        "test": "from solution import add\nassert add(1, 2) == 3",
        "language": "python3"
    }
    response = client.post("/execute/python", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_execute_python_wrong_language():
    payload = {
        "solution": "console.log('hi')",
        "test": "console.log('test')",
        "language": "javascript"
    }
    response = client.post("/execute/python", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "failed"
