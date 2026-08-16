import os
from pathlib import Path
import imageio.v2 as iio
import numpy as np
from _common import ARTIFACTS, build_cantilever_scene, probe_main

def body():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    output = ARTIFACTS / "p5_render.mp4"
    scene, rod, box, cam = build_cantilever_scene(1, with_camera=True)
    rod.attach_to_rigid_link(box.base_link, verts_ids=[0, 1])
    frames = []
    for step in range(360):
        z = 0.5 + 0.25 * min(step / 180, 1.0)
        box.set_pos(np.array([[0, 0, z]]))
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
    std = float(frame.std())
    foreground = float((frame.std(axis=-1) > 8).mean())
    assert std > 3 and foreground > 0.2, (std, foreground)
    vendor = "unverified-context-bound"
    print(f"output={output} bytes={output.stat().st_size} frames={decoded_count} std={std:.3f} fg_frac={foreground:.3f} EGL_vendor={vendor}")
    return f"mp4={output.name}; bytes={output.stat().st_size}; frames={decoded_count}; std={std:.2f}; fg={foreground:.2f}; EGL={vendor}"
if __name__ == "__main__": raise SystemExit(probe_main("p5_render", body))
