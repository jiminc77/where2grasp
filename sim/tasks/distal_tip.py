"""Distal tip placement task: settle a horizontal free arm and score its tip vs a distal target.

Reuses sim/scene.py verbatim (moving two-vertex clamp, identical terminal clamp geometry as
lift_and_clear). The ONLY new physics is the SCORING, rooted at vertex 1 (the analytic clamp
root, free length = (n_vertices-2)*interval): a two-vertex-midpoint origin would shift reach by
half an interval (5 mm) which is material at this tolerance.

  v0=verts[:,0], v1=verts[:,1] (root), tip=verts[:,-1]
  e_x = (v1-v0)/||v1-v0||          (local clamp tangent, ~ +x)
  reach = dot(tip - v1, e_x)        (horizontal extension of the tip past the root)
  droop = v1_z - tip_z              (vertical drop of the tip below the root)
  success = reach >= d-rho_x  AND  |droop-Delta| <= rho  AND  Pi_g(ell,B,w) <= 0.5
  J = rho - |droop - Delta|  when successful, else J_inf (finite pinned infeasible score)

Convergence is on the full free-vertex 3-D velocity (scene.settle); a non-converged rollout
HARD-FAILS (raises) and never enters a landscape.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import genesis as gs
from sim.scene import (add_moving_clamp, add_straight_rod, attach_moving_clamp, build_scene,
                       drive_clamp, settle, vertices)
import sim.tip_model as tm


def score_tip(state, B_eff, w, ell, Delta=tm.DELTA, rho=tm.RHO, d=tm.D, rho_x=tm.RHO_X, j_inf=tm.J_INF):
    """Vectorised root-at-vertex-1 scoring for a batched settled vertex array (n_envs, n_verts, 3)."""
    state = np.asarray(state, float)
    v0 = state[:, 0, :]; v1 = state[:, 1, :]; tip = state[:, -1, :]
    ex = v1 - v0
    ex = ex / np.maximum(np.linalg.norm(ex, axis=1, keepdims=True), 1e-12)
    reach = np.einsum('ij,ij->i', tip - v1, ex)
    droop = v1[:, 2] - tip[:, 2]
    pi = float(tm.pi_g(ell, B_eff, w))
    reach_ok = reach >= (d - rho_x)
    depth_ok = np.abs(droop - Delta) <= rho
    regime_ok = pi <= tm.PI_G_MAX
    success = reach_ok & depth_ok & bool(regime_ok)
    J = np.where(success, rho - np.abs(droop - Delta), j_inf)
    return [dict(reach=float(reach[e]), droop=float(droop[e]), pi_g=pi,
                 reach_ok=bool(reach_ok[e]), depth_ok=bool(depth_ok[e]), regime_ok=bool(regime_ok),
                 success=bool(success[e]), J=float(J[e])) for e in range(state.shape[0])]


def run_rollout(raw_E, B_eff, segment_mass, ell, n_vertices=None, template=None,
                Delta=tm.DELTA, rho=tm.RHO, d=tm.D, rho_x=tm.RHO_X, interval=0.01,
                seed=0, n_envs=1, render=False, dt=2.5e-4, substeps=20):
    """One clamp-frame distal-placement result per environment (settled cantilever sag).

    raw_E is the Genesis bending_stiffness setter knob; B_eff is the calibrated continuum
    stiffness used for scoring/Pi_g (the two are distinct -- see calibrate_beff)."""
    np.random.seed(seed)
    if n_vertices is None:
        n_vertices = int(round(ell / interval)) + 2
    template = template or {"kind": "ease", "terminal_pos": (0, 0, .7)}
    scene = build_scene(dt=dt, substeps=substeps)
    surface = (gs.surfaces.Default(
        diffuse_texture=gs.textures.ImageTexture(image_path="dlo-lab/textures/rope01.png"),
        vis_mode="recon") if render else None)
    rod = add_straight_rod(scene, n_vertices, interval=interval, E=raw_E,
                           segment_mass=segment_mass, pos=(0, 0, .5), surface=surface)
    box = add_moving_clamp(scene, pos=(0, 0, .5))
    camera = None
    if render:
        camera = scene.add_camera(res=(480, 480), pos=(.35, -.9, .55), lookat=(.25, 0, .45), fov=40, GUI=False)
    scene.build(n_envs=n_envs, env_spacing=(1.5, 1.5))
    attach_moving_clamp(rod, box)
    frames = []
    drive_clamp(scene, box, template, (0, 0, .5), steps=360)
    if camera is not None:
        frames.append(np.asarray(camera.render()[0])[..., :3])
    ok, steps, speed = settle(scene, rod, vel_tol=2e-3, window=50, max_steps=6000)
    if not ok:
        raise RuntimeError(f"distal rollout did not settle (full-3D): steps={steps} speed={speed}")
    state = vertices(rod)
    out = score_tip(state, B_eff, segment_mass * 9.81 / interval, ell, Delta, rho, d, rho_x)
    for e in out:
        e.update(ell=float(ell), converged=True, settle_steps=int(steps))
    return (out, frames) if render else out


def demo(win_cell=("B2_w1", 5621387.729022082, 0.00043088693800637695),
         fail_cell=("B1_w2", 2370946.5892385854, 0.0009283177667225556)):
    """Render a WIN (mid-rope grasp settles on the distal target) vs FAIL (soft/heavy misses) demo."""
    import imageio.v2 as iio
    root = Path(__file__).resolve().parents[1]; target = root / "figures"; target.mkdir(exist_ok=True)
    grid = tm.ell_grid()
    outcomes = []
    for name, E, mass in (win_cell, fail_cell):
        B = {2370946.5892385854: 0.01610520612280113, 5621387.729022082: 0.03825920568613013}[E]
        w = mass * 9.81 / 0.01
        a = tm.cell_analysis(B, w, grid=grid)
        ell = grid[a['argmax_idx']] if a['argmax_idx'] is not None else grid[tm.nearest_grid_index(tm.ell_optimum(B, w), grid)]
        result, frames = run_rollout(E, B, mass, float(ell), template={"kind": "ease", "terminal_pos": (0, 0, .7)}, render=True)
        # serial replay movie
        scene = build_scene(dt=2.5e-4, substeps=20)
        surface = gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path="dlo-lab/textures/rope01.png"), vis_mode="recon")
        nv = int(round(ell / 0.01)) + 2
        rod = add_straight_rod(scene, nv, interval=.01, E=E, segment_mass=mass, pos=(0, 0, .5), surface=surface)
        box = add_moving_clamp(scene, (0, 0, .5))
        cam = scene.add_camera(res=(480, 480), pos=(.35, -.9, .55), lookat=(.25, 0, .45), fov=42, GUI=False)
        scene.build(n_envs=1); attach_moving_clamp(rod, box)
        movie = []
        for i in range(480):
            t = min((i + 1) / 260, 1); z = .5 + .2 * (t * t * (3 - 2 * t)); box.set_pos(np.array([[0, 0, z]])); scene.step()
            if i % 12 == 0:
                movie.append(np.asarray(cam.render()[0])[..., :3])
        path = target / f"distal_demo_{name}.mp4"; iio.mimwrite(path, movie, fps=15, codec="libx264")
        reader = iio.get_reader(path); decoded = reader.count_frames()
        middle = np.asarray(reader.get_data(decoded // 2)); reader.close()
        std = float(middle.std()); assert decoded == len(movie) and std > 3, (decoded, len(movie), std)
        outcomes.append(dict(cell=name, ell=float(ell), reach=result[0]['reach'], droop=result[0]['droop'],
                             success=result[0]['success'], J=result[0]['J'], path=str(path),
                             bytes=path.stat().st_size, frames=decoded, frame_std=std))
    assert outcomes[0]['success'] and not outcomes[1]['success'], outcomes
    return outcomes


if __name__ == "__main__":
    print(demo())
