import ast
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

paths = [
    Path("src/llamafactory/train/trainer_utils.py"),
    Path("src/llamafactory/v1/utils/helper.py"),
]
originals = {path: path.read_text() for path in paths}
mutants = {
    "expm1_only": "return torch.log(-torch.expm1(x))",
    "unsafe_where": (
        "return torch.where(x > -math.log(2), "
        "torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))"
    ),
}
try:
    for name, expression in mutants.items():
        for path, source in originals.items():
            node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "log1mexp")
            lines = source.splitlines(keepends=True)
            replacement = lines[:node.lineno] + ["    " + expression + "\n"] + lines[node.end_lineno:]
            path.write_text("".join(replacement))
        report = Path(f"/tmp/{name}.xml")
        completed = subprocess.run([
            "uv", "run", "pytest", "-q", "--import-mode=importlib",
            "tests/train/test_log1mexp.py", "-k", "values_and_gradients",
            f"--junitxml={report}",
        ])
        suite = ET.parse(report).getroot().find("testsuite")
        counts = {key: int(suite.attrib[key]) for key in ("tests", "failures", "errors")}
        assert completed.returncode == 1 and counts == {"tests": 4, "failures": 4, "errors": 0}, (name, completed.returncode, counts)
        print(f"MUTATION {name}: all 4 numerical regression cases fail as expected", flush=True)
finally:
    for path, source in originals.items():
        path.write_text(source)
