import numpy as np
from _common import build_cantilever_scene, probe_main, vertices

def body():
    scene, rod, box, _ = build_cantilever_scene(1)
    rod.attach_to_rigid_link(box.base_link, verts_ids=[0, 1])
    for z in np.linspace(0.5, 0.8, 400):
        box.set_pos(np.array([[0, 0, z]]))
        scene.step()
    state = vertices(rod)[0]
    clamp = state[:2]
    target = np.array([0, 0, 0.8])
    tracking = float(np.linalg.norm(clamp - target, axis=1).max())
    tangent = clamp[1] - clamp[0]
    assert tracking < 0.03, tracking
    assert abs(tangent[1]) < 0.01 and abs(tangent[2]) < 0.01 and tangent[0] > 0, tangent
    assert state[-1, 2] < clamp[:, 2].mean(), (state[-1, 2], clamp[:, 2].mean())
    print(f"tracking={tracking:.6f} tangent={tangent.tolist()} tip_z={state[-1,2]:.6f}")
    return f"tracking={tracking:.5f}; tangent={tangent.tolist()}; tip hangs"
if __name__ == "__main__": raise SystemExit(probe_main("p4_clamp_lift", body))
