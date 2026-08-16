"""Live material-property control; radius and shear stiffness remain fixed."""
from __future__ import annotations

import numpy as np
import torch
import genesis as gs
from sim.scene import add_straight_rod, build_scene, settle, vertices


def apply_properties(rod, bending_stiffness, segment_mass, envs_idx=None):
    """Set batched E and per-vertex mass through Genesis' live setters."""
    E = np.atleast_1d(np.asarray(bending_stiffness, dtype=float))
    mass = np.asarray(segment_mass, dtype=float)
    if mass.ndim == 0:
        mass = np.full((E.size, rod.n_vertices), mass)
    elif mass.ndim == 1:
        mass = np.repeat(mass[:, None], rod.n_vertices, axis=1)
    assert E.ndim == 1 and mass.shape == (E.size, rod.n_vertices)
    device = "cuda"
    rod.set_bending_stiffness(torch.tensor(E, dtype=gs.tc_float, device=device), envs_idx=envs_idx)
    rod.set_segment_mass(torch.tensor(mass, dtype=gs.tc_float, device=device), envs_idx=envs_idx)
    got = rod.get_all_bending_stiffness_tc().detach().cpu().numpy().reshape(-1)
    indices = np.arange(got.size) if envs_idx is None else np.asarray(envs_idx)
    assert np.allclose(got[indices], E, rtol=1e-6, atol=1e-4), (got, E)


def stability_check(stiffest=1e8, heaviest=0.02, softest=1e6, lightest=0.002,
                    n_vertices=12, interval=0.01):
    """Exercise the full grid, using a finer integrator for the E=1e8 endpoint."""
    scene = build_scene(dt=2.5e-4, substeps=20)
    rods = [add_straight_rod(scene, n_vertices, interval, E, mass, pos=(0, .3 * i, .5))
            for i, (E, mass) in enumerate(((stiffest, heaviest), (stiffest, lightest),
                                            (softest, heaviest), (softest, lightest)))]
    scene.build(n_envs=1)
    for rod in rods:
        rod.set_fixed_states(fixed_ids=[0, 1])
    ok, steps, speed = settle(scene, rods, vel_tol=2e-2, window=30, max_steps=6000)
    finite = all(np.isfinite(vertices(rod)).all() for rod in rods)
    assert ok and finite, (ok, steps, speed, finite)
    return {"dt": 2.5e-4, "substeps": 20, "settled": ok, "steps": steps, "max_speed": speed,
            "note": "The complete calibration/task grid uses this stable integrator configuration."}
