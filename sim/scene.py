"""Genesis scene construction and cantilever measurement helpers."""
from __future__ import annotations

import numpy as np
import torch
import genesis as gs

FIXED_G = 1e4


def build_scene(dt=1e-3, substeps=5, damping=10.0, angular_damping=5.0):
    if not gs._initialized:
        gs.init(seed=0, precision="64", logging_level="warning", backend=gs.gpu)
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=dt, substeps=substeps),
                     rod_options=gs.options.RODOptions(damping=damping, angular_damping=angular_damping),
                     show_viewer=False, renderer=gs.renderers.Rasterizer())
    scene._w2g_dt = dt
    return scene


def add_straight_rod(scene, n_vertices, interval=0.01, E=1e7, segment_mass=0.02,
                     segment_radius=0.01, G=FIXED_G, pos=(0, 0, 0.5), use_inextensible=True,
                     surface=None):
    kwargs = {}
    if surface is not None:
        kwargs["surface"] = surface
    return scene.add_entity(
        material=gs.materials.ROD.Base(E=E, G=G, segment_mass=segment_mass,
                                       segment_radius=segment_radius, use_inextensible=use_inextensible),
        morph=gs.morphs.ParameterizedRod(type="rod", n_vertices=n_vertices, interval=interval,
                                         axis="x", rest_state="straight", pos=pos),
        **kwargs)


def vertices(rod):
    return rod.get_vertices_pos().detach().cpu().numpy()


def settle(scene, rods, vel_tol=1e-3, window=50, max_steps=6000):
    rods = list(rods) if isinstance(rods, (list, tuple)) else [rods]
    previous = [vertices(rod) for rod in rods]
    stable = 0
    max_speed = float("inf")
    dt = scene._w2g_dt
    for step in range(1, max_steps + 1):
        scene.step()
        current = [vertices(rod) for rod in rods]
        speeds = []
        for before, after in zip(previous, current):
            free = after[:, 2:, :] if after.shape[1] > 2 else after
            old_free = before[:, 2:, :] if before.shape[1] > 2 else before
            speeds.append(np.linalg.norm((free - old_free) / dt, axis=-1).max())
        max_speed = float(max(speeds))
        stable = stable + 1 if max_speed < vel_tol else 0
        if stable >= window:
            return True, step, max_speed
        previous = current
    return False, max_steps, max_speed


def tip_droop(rod, root_idx, tip_idx, tip_z0):
    """Downward clamp-frame tip displacement, one value per environment."""
    tip_z = vertices(rod)[:, tip_idx, 2]
    return np.asarray(tip_z0) - tip_z


def add_moving_clamp(scene, pos=(0, 0, 0.5)):
    return scene.add_entity(material=gs.materials.Rigid(needs_coup=True, coup_friction=0.9),
                            morph=gs.morphs.Box(pos=pos, size=(0.02, 0.02, 0.02), fixed=True))


def attach_moving_clamp(rod, box):
    rod.attach_to_rigid_link(box.base_link, verts_ids=[0, 1])


def drive_clamp(scene, box, template, start, steps=400):
    """Drive a box through a template; every template terminates at terminal_pos."""
    terminal = np.asarray(template["terminal_pos"], dtype=float)
    start = np.asarray(start, dtype=float)
    kind = template.get("kind", "linear")
    for i in range(steps):
        t = (i + 1) / steps
        if kind == "arc":
            p = start * (1 - t) + terminal * t
            p[0] += template.get("arc", 0.03) * np.sin(np.pi * t)
        elif kind == "ease":
            s = t * t * (3 - 2 * t)
            p = start * (1 - s) + terminal * s
        else:
            p = start * (1 - t) + terminal * t
        box.set_pos(np.asarray([p]))
        scene.step()
    return terminal
