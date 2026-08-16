"""Shared helpers for the asset-free Step-0 Genesis smoke probes."""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import genesis as gs

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
ARTIFACTS = ROOT / "artifacts"


def build_cantilever_scene(n_envs, n_vertices=40, interval=0.01, E=1e6,
                            segment_mass=0.02, segment_radius=0.01,
                            with_camera=False):
    """Build a batched rod with its first two vertices attachable to a fixed box."""
    if not gs._initialized:
        gs.init(seed=0, precision="64", logging_level="warning", backend=gs.gpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-3, substeps=5),
        rod_options=gs.options.RODOptions(damping=10.0, angular_damping=5.0),
        show_viewer=False,
        renderer=gs.renderers.Rasterizer(),
    )
    scene._smoke_dt = 1e-3
    scene.add_entity(material=gs.materials.Rigid(needs_coup=True, coup_friction=0.1),
                     morph=gs.morphs.Plane(fixed=True))
    rod = scene.add_entity(
        material=gs.materials.ROD.Base(E=E, segment_mass=segment_mass,
                                       segment_radius=segment_radius,
                                       use_inextensible=True),
        morph=gs.morphs.ParameterizedRod(type="rod", n_vertices=n_vertices,
                                         interval=interval, axis="x",
                                         rest_state="straight", pos=(0, 0, 0.5)),
    )
    box = scene.add_entity(material=gs.materials.Rigid(needs_coup=True, coup_friction=0.9),
                           morph=gs.morphs.Box(pos=(0, 0, 0.5), size=(0.02, 0.02, 0.02),
                                               fixed=True))
    cam = None
    if with_camera:
        cam = scene.add_camera(res=(480, 480), pos=(0.2, -1.0, 0.55),
                               lookat=(0.2, 0, 0.45), fov=40, GUI=False)
    scene.build(n_envs=n_envs, env_spacing=(2, 2))
    return scene, rod, box, cam


def vertices(rod):
    return rod.get_vertices_pos().detach().cpu().numpy()


def settle(scene, rod, max_steps, vel_tol, window):
    """Advance until all free vertices remain below velocity tolerance for a window."""
    previous = vertices(rod)
    stable = 0
    max_speed = float("inf")
    dt = getattr(scene, "_smoke_dt", 1e-3)
    for step in range(1, max_steps + 1):
        scene.step()
        current = vertices(rod)
        free = current[:, 2:, :] if current.shape[1] > 2 else current
        max_speed = float(np.linalg.norm((free - previous[:, 2:, :]) / dt, axis=-1).max())
        stable = stable + 1 if max_speed < vel_tol else 0
        if stable >= window:
            return True, step, max_speed
        previous = current
    return False, max_steps, max_speed


def tip_droop(rod):
    """Return clamp-height minus tip-height for every batched environment."""
    state = vertices(rod)
    return state[:, 0, 2] - state[:, -1, 2]


class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams: stream.flush()


def probe_main(name, body):
    """Execute a probe, persist its standalone log, and emit the required final line."""
    LOGS.mkdir(parents=True, exist_ok=True)
    status, detail = "PASS", ""
    with (LOGS / f"{name}.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log)), contextlib.redirect_stderr(_Tee(sys.stderr, log)):
            try:
                detail = body() or "ok"
            except Exception as exc:
                status = "FAIL"
                detail = f"{type(exc).__name__}: {exc}"
                print(detail)
            print(f"PROBE {name} {status}: {detail}")
    return 0 if status == "PASS" else 1


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
