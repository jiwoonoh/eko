"""Check README production-function coverage and local source/test anchors."""

import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def definitions(path):
    """Return qualified function names and definition lines from one Python file."""
    found = {}

    def visit(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + child.name
                found[name] = child.lineno
                visit(child, name + ".")
            elif isinstance(child, ast.ClassDef):
                visit(child, prefix + child.name + ".")
            else:
                visit(child, prefix)

    visit(ast.parse(path.read_text()))
    return found


readme = (ROOT / "README.md").read_text()
files = sorted(ROOT.glob("*.py")) + sorted((ROOT / "providers").glob("*.py"))
expected = {(str(path.relative_to(ROOT)), name): line
            for path in files for name, line in definitions(path).items()}
rows = re.findall(r"\| \[`([^`]+)`\]\(([^)#]+)#L(\d+)\) \|", readme)
actual = {(path, name): int(line) for name, path, line in rows}
assert len(rows) == len(actual), "duplicate function reference"
assert actual == expected, f"function inventory differs: {set(actual.items()) ^ set(expected.items())}"
links = 0
for target in re.findall(r"\]\(([^)]+)\)", readme):
    if "://" in target or target.startswith("#"):
        continue
    file, _, anchor = target.partition("#")
    path = ROOT / file
    assert path.exists(), target
    if anchor.startswith("L"):
        line = int(anchor[1:])
        assert 1 <= line <= len(path.read_text().splitlines()), target
        if path.suffix == ".py":
            assert line in definitions(path).values(), f"not a definition: {target}"
    links += 1
print(json.dumps({"production_functions": len(expected), "local_links_checked": links}))
