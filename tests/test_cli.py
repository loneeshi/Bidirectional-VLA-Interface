import json

import pytest

from bvi.cli import main


def test_run_defaults_to_preview_without_execution(tmp_path, capsys):
    manifest = {
        "episodes": [
            {"seed": seed, "plan_uid": f"plan-{seed % 16}", "binding": {"subtasks": [{}] * 20}}
            for seed in range(20)
        ]
    }
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps(manifest), encoding="utf-8")
    assert main([
        "run", "--setting", "fixed", "--source-manifest", str(source),
        "--output", str(tmp_path / "out"), "--checkpoint-root", str(tmp_path / "ckpt"),
    ]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["execution"] is False
    assert not (tmp_path / "out").exists()


def test_execute_requires_explicit_runtime_path(tmp_path):
    with pytest.raises(SystemExit):
        main([
            "run", "--setting", "fixed", "--source-manifest", str(tmp_path / "missing"),
            "--output", str(tmp_path / "out"), "--checkpoint-root", str(tmp_path / "ckpt"),
            "--execute",
        ])
