import json,tempfile
from pathlib import Path
from sim.history import write_provenance
p=Path(tempfile.gettempdir())/"w2g_prov.json"; o=write_provenance(p,["sim/manifests/calibration.json"]); ok=len(o["inputs"].popitem()[1])==64; print(json.dumps({"probe":"p_prov","pass":ok,"output":str(p)})); assert ok
