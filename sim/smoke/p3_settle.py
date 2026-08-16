import numpy as np
from _common import build_cantilever_scene, probe_main, settle, tip_droop, vertices

def body():
    scene, rod, _, _ = build_cantilever_scene(1)
    rod.set_fixed_states(fixed_ids=[0, 1])
    settled, steps, speed = settle(scene, rod, max_steps=4000, vel_tol=0.02, window=40)
    state = vertices(rod)
    assert settled, f"not settled after {steps}, speed={speed}"
    assert np.isfinite(state).all()
    print(f"settled={settled} steps={steps} speed={speed:.6g} tip_z={state[0,-1,2]:.6f} droop={tip_droop(rod)[0]:.6f}")
    return f"steps={steps}; speed={speed:.4g}; tip_z={state[0,-1,2]:.5f}"
if __name__ == "__main__": raise SystemExit(probe_main("p3_settle", body))
