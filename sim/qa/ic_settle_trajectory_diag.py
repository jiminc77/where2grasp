"""Diagnostic: is interval 0.01 (reference) GENUINELY unstable for option (iv), or just
under-settled? Run a long settle (up to 40000 steps) at f_eps=0.9*cap, batched 5 materials,
printing the tip-droop trajectory + tip speed so convergence vs divergence is unambiguous.
Also re-checks interval 0.005 (which showed true NaN) with the same long allowance.
"""
import numpy as np, torch, genesis as gs
from sim.scene import build_scene, add_straight_rod, vertices
from sim.calibrate_beff_force import build_mass_tensor
from sim import ic_common as ic

ELL = 0.24


def run(interval, max_steps=40000, chunk=2000):
    nv = ic.n_vertices_for(ELL, interval)
    cap = ic.max_f_eps_for_contamination(nv, interval, 0.01)
    f_eps = 0.9 * cap
    m_tips = np.array([ic.m_tip_for(b) for b in ic.GRAV_B_EFF])
    print(f"\n=== interval {interval}  nv={nv}  f_eps={f_eps:.3g} (contam {ic.contamination_ratio(nv,interval,f_eps)*100:.2f}%) ===", flush=True)
    scene = build_scene(dt=ic.DT, substeps=ic.SUBSTEPS, damping=ic.DAMPING, angular_damping=ic.ANGULAR_DAMPING)
    rod = add_straight_rod(scene, nv, interval=interval, E=1e7, segment_mass=1e-4,
                           segment_radius=ic.SEGMENT_RADIUS, G=ic.SHEAR_G, pos=(0, 0, 0.7))
    scene.build(n_envs=len(ic.RAW_E_GRID))
    rod.set_fixed_states(fixed_ids=[0, 1])
    rod.set_bending_stiffness(torch.tensor(list(ic.RAW_E_GRID), dtype=gs.tc_float, device="cuda"))
    rod.set_segment_mass(torch.tensor(build_mass_tensor(nv, m_tips, f_eps), dtype=gs.tc_float, device="cuda"))
    ti = ic.tip_index(nv)
    z0 = vertices(rod)[:, ti, 2].copy()
    prev = z0.copy()
    dt_chunk = ic.DT * chunk
    steps = 0
    while steps < max_steps:
        for _ in range(chunk):
            scene.step()
        steps += chunk
        cur = vertices(rod)[:, ti, 2].copy()
        droop = z0 - cur
        speed = np.abs(cur - prev) / dt_chunk
        prev = cur
        ratio = droop / ELL
        finite = np.isfinite(droop).all()
        print(f"  step {steps:6d}: delta/ell=[{', '.join(f'{r:+.4f}' for r in ratio)}]  "
              f"max_tip_speed={np.nanmax(speed):.2e}  finite={finite}", flush=True)
        if not finite:
            print("  -> DIVERGED (NaN/inf)", flush=True); return
        if np.nanmax(speed) < 1e-4:
            print(f"  -> SETTLED at step {steps}; delta/ell={np.array2string(ratio, precision=4)}", flush=True); return
    print(f"  -> NOT settled within {max_steps} (final max_speed shown above)", flush=True)


run(0.01, max_steps=40000)
run(0.005, max_steps=40000)
print("\n[done]", flush=True)
