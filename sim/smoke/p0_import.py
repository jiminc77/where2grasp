from pathlib import Path
import subprocess
import genesis as gs
from _common import probe_main

def body():
    assert gs.__version__ == "1.0.0", gs.__version__
    sha = subprocess.check_output(["git", "-C", str(Path.home()/"DLO-Lab"), "rev-parse", "HEAD"], text=True).strip()
    assert sha == "c5026a9416b03c6bc5186eba13cd4ffd4c0e7796", sha
    location = subprocess.check_output(["pip", "show", "genesis-world"], text=True)
    line = next(x for x in location.splitlines() if x.startswith("Location:"))
    print(f"genesis={gs.__version__} clone_sha={sha} {line}")
    return f"genesis={gs.__version__}; sha={sha}; {line}"
if __name__ == "__main__": raise SystemExit(probe_main("p0_import", body))
