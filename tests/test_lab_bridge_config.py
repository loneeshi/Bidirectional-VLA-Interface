import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('serve_vlm_bridge', ROOT / 'scripts/serve_vlm_bridge.py')
BRIDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BRIDGE)


def test_lab_credentials_loader_accepts_only_declared_fields(tmp_path):
    path = tmp_path / 'server.env'
    path.write_text('LAB_SSH_HOST=example\nLAB_SSH_PORT=22\nLAB_SSH_USER=user\n'
                    'LAB_SSH_PASSWORD="secret"\nOPENAI_API_KEY=must-not-load\n')
    loaded = BRIDGE.load_server_credentials(path)
    assert loaded == {'LAB_SSH_HOST': 'example', 'LAB_SSH_PORT': '22',
                      'LAB_SSH_USER': 'user', 'LAB_SSH_PASSWORD': 'secret'}


def test_lab_credentials_loader_requires_authentication(tmp_path):
    path = tmp_path / 'server.env'
    path.write_text('LAB_SSH_HOST=example\nLAB_SSH_PORT=22\nLAB_SSH_USER=user\n')
    with pytest.raises(ValueError, match='incomplete'):
        BRIDGE.load_server_credentials(path)
