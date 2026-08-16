from pathlib import Path
import genesis
from _common import ROOT, probe_main

def body():
    assets = Path(genesis.__file__).resolve().parent / "assets" / "dlo-lab"
    assert not assets.exists(), f"unexpected textured assets at {assets}"
    note = ROOT / "asset_note.md"
    note.write_text("""# Step-0 DLO asset status\n\n`genesis/assets/dlo-lab` is absent on this machine. The DLO-Lab textured assets are auth-gated by UMass SharePoint (HTTP 401), so textured examples are intentionally excluded from Step-0. The shipped render uses the asset-free DLO-Lab README quick-start scene (two ropes, fixed ends, and default surfaces) rendered headlessly via NVIDIA EGL.\n""", encoding="utf-8")
    print(f"expected absent assets={assets}; wrote {note}")
    return "DLO textured assets absent as expected (UMass SharePoint HTTP 401); asset-free README render documented"
if __name__ == "__main__": raise SystemExit(probe_main("asset_check", body))
