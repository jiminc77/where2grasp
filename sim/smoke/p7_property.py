import numpy as np
import torch
import genesis as gs
from _common import build_cantilever_scene, probe_main, settle, tip_droop

def settle_property(values, mass=False):
    scene, rod, _, _ = build_cantilever_scene(2, n_vertices=20, interval=0.01)
    rod.set_fixed_states(fixed_ids=[0, 1])
    val = torch.tensor(values, dtype=gs.tc_float, device="cuda")
    if mass:
        rod.set_segment_mass(val[:, None].repeat(1, 20))
    else:
        rod.set_bending_stiffness(val)
    settled, steps, speed = settle(scene, rod, 4000, 0.02, 40)
    assert settled, f"unsettled {steps}/{speed}"
    return rod, tip_droop(rod)

def body():
    stiffness = [1e7, 3e5]
    rod, stiff_droop = settle_property(stiffness)
    assert stiff_droop[0] < stiff_droop[1], stiff_droop
    getter = getattr(rod, "get_all_bending_stiffness_tc", None)
    readback = None
    if getter is not None:
        readback = getter().detach().cpu().numpy().reshape(-1)
        assert np.allclose(readback[:2], stiffness, rtol=1e-4), readback
    masses = [0.01, 0.08]
    _, mass_droop = settle_property(masses, mass=True)
    assert mass_droop[1] > mass_droop[0], mass_droop
    print(f"stiff_droop={stiff_droop.tolist()} mass_droop={mass_droop.tolist()} getter={None if readback is None else readback.tolist()}")
    return f"stiff={stiff_droop.tolist()}; mass={mass_droop.tolist()}; getter={'checked' if readback is not None else 'unavailable'}"
if __name__ == "__main__": raise SystemExit(probe_main("p7_property", body))
