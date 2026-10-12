import io
from pathlib import Path
import sys
from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest

# Ensure repository root is on sys.path for test runners
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legacy_scripts.server_failed import TranscribeResponse, app


@pytest.fixture
def client():
    return TestClient(app)


def test_path_traversal_rejected(client):
    """Path traversal in filename must be rejected with HTTP 400."""
    response = client.post(
        "/transcribe",
        files={"file": ("../../etc/passwd.wav", io.BytesIO(b"RIFFdummydata"), "audio/wav")}
    )
    assert response.status_code == 400
    assert "traversal" in response.json()["detail"].lower() or "invalid" in response.json()["detail"].lower()


def test_disallowed_extension_rejected(client):
    """Unsupported file extensions must be rejected with HTTP 400."""
    response_exe = client.post(
        "/transcribe",
        files={"file": ("malware.exe", io.BytesIO(b"MZdummy"), "application/octet-stream")}
    )
    assert response_exe.status_code == 400
    assert "unsupported audio format" in response_exe.json()["detail"].lower()

    response_txt = client.post(
        "/transcribe",
        files={"file": ("notes.txt", io.BytesIO(b"text data"), "text/plain")}
    )
    assert response_txt.status_code == 400
    assert "unsupported audio format" in response_txt.json()["detail"].lower()


def test_transcribe_response_schema_success(client):
    """Valid audio processing must return validated TranscribeResponse schema."""
    with patch("legacy_scripts.server_failed._run_inference") as mock_inference:
        mock_inference.return_value = (
            [{"Start": 0.0, "End": 5.0, "Speaker": "1", "Content": "Test transcription"}],
            False
        )
        response = client.post(
            "/transcribe",
            files={"file": ("sample.wav", io.BytesIO(b"RIFFvalidbytes"), "audio/wav")}
        )

        assert response.status_code == 200
        data = response.json()
        validated = TranscribeResponse(**data)
        assert validated.filename == "sample.wav"
        assert validated.status == "success"
        assert validated.is_fallback is False
        assert validated.warning is None
        assert len(validated.transcription) == 1
        assert validated.transcription[0].Content == "Test transcription"
        assert validated.transcription[0].Start == 0.0
        assert validated.transcription[0].End == 5.0


def test_transcribe_response_schema_fallback(client):
    """Fallback recovery must populate is_fallback and warning in TranscribeResponse."""
    with patch("legacy_scripts.server_failed._run_inference") as mock_inference:
        mock_inference.return_value = (
            [{"Start": 0.0, "End": 0.0, "Speaker": "Unknown", "Content": "Raw fallback text"}],
            True
        )
        response = client.post(
            "/transcribe",
            files={"file": ("sample.mp3", io.BytesIO(b"ID3dummydata"), "audio/mpeg")}
        )

        assert response.status_code == 200
        data = response.json()
        validated = TranscribeResponse(**data)
        assert validated.filename == "sample.mp3"
        assert validated.is_fallback is True
        assert validated.warning is not None
        assert "token limit" in validated.warning.lower()
        assert len(validated.transcription) == 1
        assert validated.transcription[0].Speaker == "Unknown"
        assert validated.transcription[0].Content == "Raw fallback text"
