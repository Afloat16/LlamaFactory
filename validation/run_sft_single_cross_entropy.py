import json
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

BASE = "6d1059c727450a4f58e56ad204d01a71abad7781"
HEAD = "466428c0fcbcf0289524bde8aba6a0077ccf103f"
SOURCE = "src/llamafactory/v1/core/base_trainer.py"
TEST = "tests_v1/trainers/test_sft_logprob_forward.py"
for path in (SOURCE, TEST):
    assert Path(path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"]), path

def run(label, tests):
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-s", *tests, f"--junitxml={label}.xml"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    Path(f"{label}.log").write_text(result.stdout)
    print(result.stdout, flush=True)
    root = ElementTree.parse(f"{label}.xml").getroot()
    counts = {key: sum(int(s.get(key, 0)) for s in root.iter("testsuite"))
              for key in ("tests", "failures", "errors", "skipped")}
    counts["returncode"] = result.returncode
    print(label, counts, flush=True)
    return counts

fixed = Path(SOURCE).read_bytes()
try:
    Path(SOURCE).write_bytes(subprocess.check_output(["git", "show", f"{BASE}:{SOURCE}"]))
    before = run("before", [TEST])
finally:
    Path(SOURCE).write_bytes(fixed)
after = run("after", [TEST])
assert before == {"tests": 14, "failures": 12, "errors": 0, "skipped": 0, "returncode": 1}, before
assert after == {"tests": 14, "failures": 0, "errors": 0, "skipped": 0, "returncode": 0}, after
existing = run("existing", ["tests_v1/trainers/test_dpo_loss_precision.py", "tests_v1/plugins/model_plugins/test_chunk_loss.py"])
assert existing["returncode"] == existing["failures"] == existing["errors"] == 0, existing
for path in (SOURCE, TEST):
    assert Path(path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"]), path
result = {"base": BASE, "head": HEAD, "before": before, "after": after, "existing": existing}
Path("validation_result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
