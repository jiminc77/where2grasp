import numpy as np
import torch
import genesis as gs
from _common import build_cantilever_scene, probe_main, settle, tip_droop, vertices

def run_batch(n_envs, stiffness):
    scene, rod, _, _ = build_cantilever_scene(n_envs)
    rod.set_fixed_states(fixed_ids=[0, 1])
    rod.set_bending_stiffness(torch.tensor(stiffness, dtype=gs.tc_float, device="cuda"))
    settled, steps, speed = settle(scene, rod, 4500, 0.025, 40)
    assert settled, f"unsettled: {steps} {speed}"
    assert np.isfinite(vertices(rod)).all()
    return tip_droop(rod), steps

def body():
    torch.cuda.reset_peak_memory_stats()
    stiffness = [3e5, 1e6, 3e6, 1e7]
    droops, steps = run_batch(4, stiffness)
    assert np.all(np.diff(droops) <= 0.003), droops
    serial, serial_steps = run_batch(1, [stiffness[0]])
    delta = abs(float(droops[0] - serial[0]))
    assert delta < 0.02, (droops[0], serial[0], delta)
    vram = torch.cuda.max_memory_allocated()
    print(f"stiffness={stiffness} droops={droops.tolist()} batch_steps={steps} serial_steps={serial_steps} delta={delta:.6f} vram={vram}")
    return f"droops={droops.tolist()}; serial_delta={delta:.5f}; vram={vram}"
if __name__ == "__main__": raise SystemExit(probe_main("p6_batch", body))
