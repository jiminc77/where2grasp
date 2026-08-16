import json,numpy as np
from sim.history import supported_wrench
from sim.sweep import settings
m=json.load(open("sim/manifests/sweep_manifest.json")); s=settings(m); vals=[supported_wrench(mass,14,m["interval"],m["gravity"]) for mass in m["segment_masses"]]; ok=np.isfinite(vals).all() and np.all(np.diff(vals)>0); print(json.dumps({"probe":"p_w","pass":bool(ok),"normalized_Fz":vals,"bending_force_diagnostic":"Genesis internal bending force is qualitative clamp cross-check only; production channel is supported weight."})); assert ok
