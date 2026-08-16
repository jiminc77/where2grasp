"""DLO-Lab README quick-start scene (asset-free), rendered headless via NVIDIA EGL."""
from pathlib import Path
import imageio.v2 as iio
import numpy as np
import genesis as gs
from _common import ARTIFACTS

def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    output = ARTIFACTS / "step0_shipped_example.mp4"
    gs.init(seed=0, precision="64", logging_level="warning", backend=gs.gpu)
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1e-3, substeps=5),
                     rod_options=gs.options.RODOptions(damping=10.0, angular_damping=5.0),
                     show_viewer=False, renderer=gs.renderers.Rasterizer())
    scene.add_entity(material=gs.materials.Rigid(needs_coup=True, coup_friction=0.1),
                     morph=gs.morphs.Plane(fixed=True))
    v1 = scene.add_entity(material=gs.materials.ROD.Base(segment_radius=0.005, E=1e5, G=1e4),
                          morph=gs.morphs.ParameterizedRod(type="rod", n_vertices=100, interval=0.01,
                          axis="x", pos=(0.5, 0.5, 0.3), euler=(0.0, 0.0, 15.0)))
    v2 = scene.add_entity(material=gs.materials.ROD.Base(segment_radius=0.005, E=1e5, G=1e4),
                          morph=gs.morphs.ParameterizedRod(type="rod", n_vertices=80, interval=0.01,
                          axis="x", pos=(0.55, 0.43, 0.4), euler=(0.0, 0.0, 0.0)))
    b1 = scene.add_entity(material=gs.materials.ROD.Base(segment_radius=0.02),
                          morph=gs.morphs.ParameterizedRod(type="rod", n_vertices=3, interval=0.1,
                          axis="x", pos=(0.75, 0.435, 0.25), euler=(0.0, 0.0, -75.0)))
    b2 = scene.add_entity(material=gs.materials.ROD.Base(segment_radius=0.02),
                          morph=gs.morphs.ParameterizedRod(type="rod", n_vertices=3, interval=0.1,
                          axis="x", pos=(1.05, 0.435, 0.25), euler=(0.0, 0.0, -75.0)))
    cam = scene.add_camera(res=(480, 480), pos=(1.0, -1.3, 0.85), lookat=(1.0, 0.45, 0.2), fov=45, GUI=False)
    scene.build(n_envs=1)
    v1.set_fixed_states(fixed_ids=[0, 1]); v2.set_fixed_states(fixed_ids=[78, 79])
    b1.set_fixed_states(fixed_ids=[0, 1, 2]); b2.set_fixed_states(fixed_ids=[0, 1, 2])
    frames = []
    for step in range(840):
        scene.step()
        if step % 12 == 0:
            frames.append(np.asarray(cam.render()[0])[..., :3])
    iio.mimwrite(output, frames, fps=15, codec="libx264")
    assert output.exists() and output.stat().st_size > 1000
    reader = iio.get_reader(output)
    decoded_count = reader.count_frames()
    assert decoded_count == len(frames), (decoded_count, len(frames))
    frame = np.asarray(reader.get_data(decoded_count // 2))
    reader.close()
    std = float(frame.std()); foreground = float((frame.std(axis=-1) > 8).mean())
    assert std > 3 and foreground > 0.2, (std, foreground)
    print(f"SHIPPED_EXAMPLE PASS: output={output} bytes={output.stat().st_size} frames={decoded_count} std={std:.3f} fg_frac={foreground:.3f}")
if __name__ == "__main__": main()
