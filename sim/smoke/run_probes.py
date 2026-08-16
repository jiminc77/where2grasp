"""Run each Step-0 smoke probe in an isolated Python process."""
import os
import re
import subprocess
import sys
from pathlib import Path
from _common import LOGS, ROOT, write_json

PROBES = ["p0_import", "p1_backend", "p2_coldjit", "p3_settle", "p4_clamp_lift", "p5_render", "p6_batch", "p7_property", "asset_check"]

def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    results = []
    for name in PROBES:
        result = subprocess.run([sys.executable, str(ROOT / f"{name}.py")], cwd=ROOT,
                                text=True, capture_output=True, env=os.environ.copy())
        text = result.stdout + result.stderr
        (LOGS / f"{name}.log").write_text(text, encoding="utf-8")
        matches = re.findall(rf"PROBE {re.escape(name)} (PASS|FAIL): (.*)", text)
        status, detail = matches[-1] if matches else ("FAIL", f"exit={result.returncode}; missing final PROBE line")
        if result.returncode and status == "PASS":
            status, detail = "FAIL", f"exit={result.returncode}; {detail}"
        results.append({"probe": name, "status": status, "detail": detail})
        print(f"PROBE {name} {status}: {detail}")
    write_json(ROOT / "probe_results.json", results)
    return 0 if all(item["status"] == "PASS" for item in results) else 1

if __name__ == "__main__": raise SystemExit(main())
