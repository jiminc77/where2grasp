import numpy as np
from sim.calibrate_beff import through_origin
from sim.material import apply_properties
from sim.scene import add_straight_rod, build_scene, settle, vertices
from sim.tasks.lift_and_clear import run_rollout


def test_unit_calibration_recovery():
    rng=np.random.default_rng(4); B=3.2e5; ell=np.linspace(.03,.08,9); w=.15
    delta=w*ell**4/(8*B)*(1+rng.normal(0,.01,len(ell)))
    slope,pred,rel=through_origin(w*ell**4,delta); recovered=1/(8*slope)
    per=w*ell**4/(8*delta); cv=per.std(ddof=1)/per.mean()
    assert abs(recovered-B)/B < .05 and rel.max() < .05 and cv < .05


def test_unit_setter_readback():
    scene=build_scene(); rod=add_straight_rod(scene,10); scene.build(n_envs=2)
    apply_properties(rod,[1e6,2e6],[.001,.002])
    got=rod.get_all_bending_stiffness_tc().detach().cpu().numpy().reshape(-1)
    assert np.allclose(got,[1e6,2e6])


def test_unit_monotonicity():
    # This uses the same batched live-property path as the validated Genesis probe.
    scene=build_scene(); rod=add_straight_rod(scene,20,E=1e6,segment_mass=.02); scene.build(n_envs=2)
    rod.set_fixed_states(fixed_ids=[0,1])
    apply_properties(rod,[1e7,3e5],[.02,.02])
    ok,_,_=settle(scene,rod,vel_tol=.02,window=40); assert ok
    stiff=[.5-vertices(rod)[e,-1,2] for e in range(2)]
    scene=build_scene(); rod=add_straight_rod(scene,20,E=1e6,segment_mass=.02); scene.build(n_envs=2)
    rod.set_fixed_states(fixed_ids=[0,1])
    apply_properties(rod,[1e6,1e6],[.01,.08])
    ok,_,_=settle(scene,rod,vel_tol=.02,window=40); assert ok
    heavy=[.5-vertices(rod)[e,-1,2] for e in range(2)]
    assert stiff[0] < stiff[1] and heavy[0] < heavy[1],(stiff,heavy)


def test_integration_label_flip_and_clamp_frame():
    common=dict(bending_stiffness=1e7,segment_mass=.001,n_vertices=14,template={"kind":"linear","terminal_pos":(0,0,.7)})
    baseline=run_rollout(**common,h=1)[0]; d=baseline['delta_tip']
    low=run_rollout(**common,h=d*.5)[0]; high=run_rollout(**common,h=d*1.5)[0]
    assert not low['success'] and high['success']
    assert abs(baseline['delta_tip']-(baseline['clamp_z']-baseline['settled_tip_z']))<1e-9
