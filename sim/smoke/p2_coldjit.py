import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from _common import probe_main

CHILD = '''import numpy as np\nimport genesis as gs\ngs.init(seed=0, precision="64", logging_level="info", backend=gs.gpu)\nscene=gs.Scene(sim_options=gs.options.SimOptions(dt=1e-3, substeps=5), rod_options=gs.options.RODOptions(damping=10.0, angular_damping=5.0), show_viewer=False)\nscene.add_entity(material=gs.materials.Rigid(needs_coup=True), morph=gs.morphs.Plane(fixed=True))\nrod=scene.add_entity(material=gs.materials.ROD.Base(E=1e6, segment_mass=0.02, segment_radius=0.01, use_inextensible=True), morph=gs.morphs.ParameterizedRod(type="rod", n_vertices=40, interval=0.01, axis="x", rest_state="straight", pos=(0,0,0.5)))\nscene.build(n_envs=1, env_spacing=(2,2))\nscene.step()\nx=rod.get_vertices_pos().detach().cpu().numpy()\nassert np.isfinite(x).all()\nprint("FINITE_STATE", x[0,-1].tolist())\n'''

def run(cache):
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(cache)
    start = time.monotonic()
    result = subprocess.run([sys.executable, "-c", CHILD], text=True, capture_output=True, env=env)
    return result, time.monotonic() - start

def body():
    cache = Path(tempfile.mkdtemp(prefix="w2g_coldcache_"))
    try:
        cold, cold_time = run(cache)
        warm, warm_time = run(cache)
        combined = cold.stdout + cold.stderr
        assert cold.returncode == 0 and "FINITE_STATE" in combined, combined[-1000:]
        assert warm.returncode == 0 and "FINITE_STATE" in (warm.stdout + warm.stderr), (warm.stdout + warm.stderr)[-1000:]
        compiled = "Compiling simulation kernels" in combined
        cache_observed = any(cache.rglob("*"))
        evidence = "compiled" if compiled else "UNVERIFIED"
        print(f"cold_time={cold_time:.3f}s warm_time={warm_time:.3f}s cold_compile={evidence} cache_populated={cache_observed}")
        if not compiled:
            print("Cold cache redirect/compiler log was not observable; functional cold/warm build verified.")
        return f"cold={cold_time:.3f}s; warm={warm_time:.3f}s; cold_compile={evidence}; cache_populated={cache_observed}"
    finally:
        shutil.rmtree(cache, ignore_errors=True)
if __name__ == "__main__": raise SystemExit(probe_main("p2_coldjit", body))
