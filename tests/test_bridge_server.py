import os

from bvi.bridge_server import load_local_credentials, load_server_credentials


def test_provider_loader_ignores_unapproved_environment_keys(tmp_path, monkeypatch):
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "UNRELATED_SECRET"):
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / ".env.local"
    path.write_text(
        "OPENAI_API_KEY=provider-key\nUNRELATED_SECRET=must-not-load\nLAB_SSH_PASSWORD=also-not-load\n",
        encoding="utf-8",
    )
    load_local_credentials(path)
    assert os.environ["OPENAI_API_KEY"] == "provider-key"
    assert "UNRELATED_SECRET" not in os.environ
    assert "LAB_SSH_PASSWORD" not in os.environ


def test_server_loader_returns_only_declared_fields(tmp_path):
    path = tmp_path / "lab.env"
    path.write_text(
        "LAB_SSH_HOST=example\nLAB_SSH_PORT=22\nLAB_SSH_USER=user\n"
        "LAB_SSH_PASSWORD=fixture-secret\nOPENAI_API_KEY=must-not-load\n",
        encoding="utf-8",
    )
    values = load_server_credentials(path)
    assert values == {
        "LAB_SSH_HOST": "example",
        "LAB_SSH_PORT": "22",
        "LAB_SSH_USER": "user",
        "LAB_SSH_PASSWORD": "fixture-secret",
    }
