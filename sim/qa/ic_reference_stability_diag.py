"""Definitive epsilon stable-window search at the REFERENCE interval 0.01.

Batches ONE material across many f_eps values in a single settle (each env a different
f_eps). Spans below the contamination cap AND well above it, to locate the numerical
stability threshold f_eps* and show whether ANY admissible (contamination<1%) f_eps is
stable. Stability = long settle reaches a small tip speed with a sensible delta/ell.
Repeats for the softest and stiffest materials (extreme mass ratios).
"""
import numpy as np, torch, genesis as gs
from sim.scene import build_scene, add_straight_rod, vertices
from sim import ic_common as ic

INTERVAL = 0.01
ELL = 0.24
NV = ic.n_vertices_for(ELL, INTERVAL)
CAP = ic.max_f_eps_for_contamination(NV, INTERVAL, 0.01)
# f_eps candidates spanning below cap -> far above cap (admissible only if <= CAP)
FEPS = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]) * CAP
MATS = {"softest": 0, "stiffest": 4}


def search(mat_name, mi, max_steps=24000, chunk=2000):
    raw_e = ic.RAW_E_GRID[mi]
    m_tip = ic.m_tip_for(ic.GRAV_B_EFF[mi])
    print(f"\n=== {mat_name}: raw_E={raw_e:.3g} m_tip={m_tip:.4g}  interval={INTERVAL} nv={NV}  "
          f"contam_cap f_eps={CAP:.3g} (arm_mass_cap={CAP*m_tip:.3g}) ===", flush=True)
    scene = build_scene(dt=ic.DT, substeps=ic.SUBSTEPS, damping=ic.DAMPING, angular_damping=ic.ANGULAR_DAMPING)
    rod = add_straight_rod(scene, NV, interval=INTERVAL, E=1e7, segment_mass=1e-4,
                           segment_radius=ic.SEGMENT_RADIUS, G=ic.SHEAR_G, pos=(0, 0, 0.7))
    scene.build(n_envs=len(FEPS))
    rod.set_fixed_states(fixed_ids=[0, 1])
    rod.set_bending_stiffness(torch.tensor([raw_e] * len(FEPS), dtype=gs.tc_float, device="cuda"))
    mass = np.full((len(FEPS), NV), 0.0)
    for e, fe in enumerate(FEPS):
        mass[e, :] = fe * m_tip
        mass[e, ic.tip_index(NV)] = m_tip
    rod.set_segment_mass(torch.tensor(mass, dtype=gs.tc_float, device="cuda"))
    ti = ic.tip_index(NV)
    z0 = vertices(rod)[:, ti, 2].copy()
    prev = z0.copy()
    dtc = ic.DT * chunk
    steps = 0
    settled = {}
    while steps < max_steps:
        for _ in range(chunk):
            scene.step()
        steps += chunk
        cur = vertices(rod)[:, ti, 2].copy()
        speed = np.abs(cur - prev) / dtc
        prev = cur
        for e in range(len(FEPS)):
            if e not in settled and np.isfinite(cur[e]) and speed[e] < 1e-4:
                settled[e] = (steps, (z0[e] - cur[e]) / ELL)
    cur = vertices(rod)[:, ti, 2].copy()
    for e, fe in enumerate(FEPS):
        cont = ic.contamination_ratio(NV, INTERVAL, fe)
        adm = "ADMISSIBLE" if fe <= CAP else "inadmissible(contam>1%)"
        if e in settled:
            st, ratio = settled[e]
            print(f"  f_eps={fe:.3g} ({fe/CAP:.1f}x cap, contam={cont*100:.2f}%, {adm}): "
                  f"SETTLED@{st} delta/ell={ratio:+.4f}", flush=True)
        else:
            fin = np.isfinite(cur[e])
            print(f"  f_eps={fe:.3g} ({fe/CAP:.1f}x cap, contam={cont*100:.2f}%, {adm}): "
                  f"NOT settled (finite={fin})", flush=True)
    stable_feps = [FEPS[e] for e in settled]
    if stable_feps:
        thr = min(stable_feps)
        print(f"  -> stability threshold f_eps* ~ {thr:.3g} = {thr/CAP:.1f}x the contamination cap "
              f"-> {'ADMISSIBLE window NON-EMPTY' if thr <= CAP else 'admissible window EMPTY (f_eps* > cap)'}",
              flush=True)
    else:
        print("  -> NO f_eps settled even far above the cap (need larger arm mass / different method)", flush=True)


for name, mi in MATS.items():
    search(name, mi)
print("\n[done]", flush=True)
