"""Confirm interval 0.005 is GENUINELY non-viable for baseline subtraction (not merely
under-settled) by running a long settle (up to 60000 steps) for the stiffest + a mid
material and printing the residual trajectory. More settle steps do NOT change the frozen
integrator (dt/substeps); they only allow more time to reach equilibrium.
"""
import numpy as np, torch, genesis as gs
from sim.scene import build_scene, add_straight_rod, vertices
from sim.calibrate_beff_force import build_baseline_tensors
from sim import ic_common as ic

INTERVAL, ELL = 0.005, 0.24
NV = ic.n_vertices_for(ELL, INTERVAL)
MATS = {"B4_stiffest": 4, "B2_mid": 2}


def check(name, mi):
    m_load = ic.m_tip_for(ic.GRAV_B_EFF[mi])
    print(f"\n=== {name} interval={INTERVAL} nv={NV} m_load={m_load:.4g} ===", flush=True)
    scene = build_scene(dt=ic.DT, substeps=ic.SUBSTEPS, damping=ic.DAMPING, angular_damping=ic.ANGULAR_DAMPING)
    rod = add_straight_rod(scene, NV, interval=INTERVAL, E=1e7, segment_mass=ic.BASELINE_ARM_MASS,
                           segment_radius=ic.SEGMENT_RADIUS, G=ic.SHEAR_G, pos=(0, 0, 0.7))
    scene.build(n_envs=1)
    rod.set_fixed_states(fixed_ids=[0, 1])
    rod.set_bending_stiffness(torch.tensor([ic.RAW_E_GRID[mi]], dtype=gs.tc_float, device="cuda"))
    baseline, loaded = build_baseline_tensors(NV, [m_load], ic.BASELINE_ARM_MASS)
    ti = ic.tip_index(NV)
    z0 = vertices(rod)[0, ti, 2]
    # settle baseline then loaded, print trajectory of the loaded phase (the harder one)
    rod.set_segment_mass(torch.tensor(loaded, dtype=gs.tc_float, device="cuda"))
    prev = vertices(rod)[0, ti, 2]
    for c in range(30):     # 30 x 2000 = 60000 steps
        for _ in range(2000):
            scene.step()
        cur = vertices(rod)[0, ti, 2]
        speed = abs(cur - prev) / (ic.DT * 2000)
        ratio = (z0 - cur) / ELL if np.isfinite(cur) else float("nan")
        prev = cur
        if c % 3 == 0 or not np.isfinite(cur):
            print(f"  step {2000*(c+1):6d}: delta_total/ell={ratio:+.4g}  tip_speed={speed:.2e}  finite={np.isfinite(cur)}", flush=True)
        if not np.isfinite(cur):
            print("  -> DIVERGED (NaN)", flush=True); return
        if speed < 1e-4:
            print(f"  -> SETTLED@{2000*(c+1)} delta_total/ell={ratio:+.4f}", flush=True); return
    print("  -> NOT settled within 60000 (see trajectory)", flush=True)


for n, i in MATS.items():
    check(n, i)
print("\n[done]", flush=True)
