"""C0 scoring unit tests (pure; synthetic vertex arrays, no Genesis).

Root-at-vertex-1 frame (HB-A7): straight-rod reach==ell, rigid-translation invariance,
vertical sign, batched shapes, last-vertex tip, J vs J_inf, in-regime/out-of-regime gating.
"""
from __future__ import annotations
import numpy as np
import sim.tip_model as tm
from sim.tasks.distal_tip import score_tip

INTERVAL = 0.01


def _straight(n, droop_tip=0.0, offset=(0., 0., 0.), interval=INTERVAL):
    """Synthetic straight rod along +x with n vertices; optional linear tip droop and rigid offset."""
    xs = np.arange(n) * interval
    z = np.zeros(n)
    if droop_tip:                                  # quadratic sag reaching droop_tip at the tip
        free = np.clip(np.arange(n) - 1, 0, None)
        z = -droop_tip * (free / max(free.max(), 1)) ** 2
    v = np.stack([xs, np.zeros(n), z], axis=1) + np.asarray(offset)
    return v[None, :, :]                            # (1, n, 3)


def test_straight_rod_reach_equals_ell():
    n = 22                                          # free length ell = (n-2)*interval = 0.20
    ell = (n - 2) * INTERVAL
    B, w = 0.09110646283229176, 0.1962
    out = score_tip(_straight(n), B, w, ell)[0]
    assert abs(out['reach'] - ell) < 1e-9
    assert abs(out['droop']) < 1e-12


def test_rigid_translation_invariance():
    n = 30; ell = (n - 2) * INTERVAL
    B, w = 0.09110646283229176, 0.1962
    a = score_tip(_straight(n, droop_tip=0.02), B, w, ell)[0]
    b = score_tip(_straight(n, droop_tip=0.02, offset=(1.3, -0.7, 2.1)), B, w, ell)[0]
    assert abs(a['reach'] - b['reach']) < 1e-9 and abs(a['droop'] - b['droop']) < 1e-9


def test_vertical_sign():
    n = 26; ell = (n - 2) * INTERVAL
    out = score_tip(_straight(n, droop_tip=0.03), 0.09110646283229176, 0.1962, ell)[0]
    assert out['droop'] > 0                          # tip below root -> positive droop


def test_batched_shapes_and_last_vertex_tip():
    n = 24; ell = (n - 2) * INTERVAL
    batch = np.concatenate([_straight(n, droop_tip=0.01), _straight(n, droop_tip=0.05)], axis=0)
    outs = score_tip(batch, 0.09110646283229176, 0.1962, ell)
    assert len(outs) == 2 and outs[1]['droop'] > outs[0]['droop']
    # last-vertex selection: moving only the last vertex changes reach/droop
    moved = _straight(n, droop_tip=0.0); moved[0, -1, 0] += 0.05
    assert score_tip(moved, 0.09110646283229176, 0.1962, ell)[0]['reach'] > ell


def test_J_and_success_in_regime():
    # choose ell/material so the grasp is in-regime and depth ~ Delta => success, J = rho-|droop-Delta|
    B, w = 0.09110646283229176, 0.1962
    ell = 0.30
    droop = tm.DELTA                                 # exactly on target depth
    n = int(round(ell / INTERVAL)) + 2
    v = _straight(n); v[0, -1, 2] = v[0, 1, 2] - droop   # set tip droop exactly
    out = score_tip(v, B, w, ell)[0]
    assert out['regime_ok'] and out['depth_ok']
    if out['reach_ok']:
        assert out['success'] and abs(out['J'] - (tm.RHO - abs(droop - tm.DELTA))) < 1e-9


def test_out_of_regime_forces_J_inf():
    # a long soft arm is out of regime (Pi_g > 0.5) -> J pinned to J_inf regardless of reach/depth
    B, w = 0.01610520612280113, 0.9106797291548271
    ell = 0.40                                       # Pi_g = w*ell^3/B huge
    assert tm.pi_g(ell, B, w) > 0.5
    n = int(round(ell / INTERVAL)) + 2
    v = _straight(n); v[0, -1, 2] = v[0, 1, 2] - tm.DELTA
    out = score_tip(v, B, w, ell)[0]
    assert not out['regime_ok'] and out['J'] == tm.J_INF and not out['success']
