"""Horizontal two-vertex-clamp lift-and-clear diagnostic."""
from __future__ import annotations
from pathlib import Path
import imageio.v2 as iio
import numpy as np
import genesis as gs
from sim.scene import add_moving_clamp, add_straight_rod, attach_moving_clamp, build_scene, drive_clamp, settle, vertices


def run_rollout(bending_stiffness, segment_mass, n_vertices=20, grasp_offset=0.0,
                template=None, h=.03, seed=0, n_envs=1, render=False):
    """Return one clamp-frame absolute-clearance result per environment."""
    np.random.seed(seed)
    template = template or {"kind": "linear", "terminal_pos": (grasp_offset, 0, .7)}
    scene = build_scene(dt=2.5e-4, substeps=20)
    surface = (gs.surfaces.Default(
        diffuse_texture=gs.textures.ImageTexture(image_path="dlo-lab/textures/rope01.png"),
        vis_mode="recon") if render else None)
    rod = add_straight_rod(scene, n_vertices, interval=.01, E=bending_stiffness,
                           segment_mass=segment_mass, pos=(grasp_offset, 0, .5), surface=surface)
    box = add_moving_clamp(scene, pos=(grasp_offset, 0, .5))
    camera = None
    if render:
        camera = scene.add_camera(res=(480,480), pos=(.15,-.9,.58), lookat=(.1,0,.48), fov=38, GUI=False)
    scene.build(n_envs=n_envs, env_spacing=(1.5, 1.5))
    attach_moving_clamp(rod, box)
    frames=[]
    drive_clamp(scene, box, template, (grasp_offset,0,.5), steps=360)
    if camera is not None: frames.append(np.asarray(camera.render()[0])[..., :3])
    ok, _, _ = settle(scene, rod, vel_tol=2e-3, window=50, max_steps=6000)
    assert ok, "rollout did not settle"
    state=vertices(rod)
    clamp_z=state[:, :2, 2].mean(axis=1)
    delta=clamp_z-state[:, -1, 2]  # invariant to rigid translation of the clamp
    out=[dict(settled_tip_z=float(state[e,-1,2]), clamp_z=float(clamp_z[e]), delta_tip=float(delta[e]),
              J=float(h-delta[e]), success=bool(delta[e] <= h)) for e in range(n_envs)]
    return (out, frames) if render else out


def demo():
    root=Path(__file__).resolve().parents[1]; target=root/"figures"; target.mkdir(exist_ok=True)
    template={"kind":"ease", "terminal_pos":(0,0,.7)}; h=.01
    # Materials selected from the stable raw-E range; same terminal geometry.
    outcomes=[]
    for name,E,mass in (("stiff",1e7,.001),("soft",1e6,.001)):
        result, frames=run_rollout(E,mass,n_vertices=20,template=template,h=h,render=True)
        # A serial replay creates a visually useful trajectory movie.
        scene=build_scene(dt=2.5e-4, substeps=20)
        surface=gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path="dlo-lab/textures/rope01.png"), vis_mode="recon")
        rod=add_straight_rod(scene,20,interval=.01,E=E,segment_mass=mass,pos=(0,0,.5),surface=surface); box=add_moving_clamp(scene,(0,0,.5))
        cam=scene.add_camera(res=(480,480),pos=(.12,-.85,.6),lookat=(.1,0,.5),fov=40,GUI=False); scene.build(n_envs=1); attach_moving_clamp(rod,box)
        movie=[]
        for i in range(480):
            t=min((i+1)/260,1); z=.5+.2*(t*t*(3-2*t)); box.set_pos(np.array([[0,0,z]])); scene.step()
            if i%12==0: movie.append(np.asarray(cam.render()[0])[..., :3])
        path=target/f"demo_{name}.mp4"; iio.mimwrite(path,movie,fps=15,codec="libx264")
        reader=iio.get_reader(path); decoded=reader.count_frames()
        middle=np.asarray(reader.get_data(decoded//2)); reader.close()
        std=float(middle.std()); assert decoded == len(movie) and std > 3, (decoded,len(movie),std)
        outcomes.append((name,result[0],str(path),path.stat().st_size,{"frames":decoded,"frame_std":std}))
    assert outcomes[0][1]["success"] and not outcomes[1][1]["success"], outcomes
    return outcomes

if __name__ == "__main__": print(demo())
