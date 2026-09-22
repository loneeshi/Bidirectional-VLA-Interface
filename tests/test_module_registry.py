import ast
from pathlib import Path
import re


def test_module_registry_matches_files_and_docstrings():
    root = Path(__file__).parents[1] / 'src'
    files = {p.relative_to(root).as_posix(): p for p in (root/'bvi').rglob('*.py')}
    index = (root/'README.md').read_text(encoding='utf-8')
    rows = re.findall(r'\| `(bvi/[^`]+\.py)` \| (active|frozen) \|', index)
    assert len(rows) == len(files)
    assert set(dict(rows)) == set(files)
    for name, status in rows:
        doc = ast.get_docstring(ast.parse(files[name].read_text(encoding='utf-8')))
        assert doc and f'STATUS: {status}' in doc
