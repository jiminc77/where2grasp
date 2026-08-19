"""C0 tests for the Item-1 k-interaction adaptation curve (sim/adaptation_curve.py).

Pure-math / synthetic / prior-committed-fixture only. NO GPU sweep on the cohort.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from sim import adaptation_curve as ac

MAN = Path(__file__).resolve().parents[1] / "manifests"

# the plan's frozen enumerated schedules (byte-match targets)
S_LIFT_CELLS = [0, 7, 14, 4, 11, 1, 8, 15, 5, 12, 2, 9, 16, 6, 13, 3, 10,
                0, 7, 14, 4, 11, 1, 8, 15, 5, 12, 2, 9, 16, 6, 13]
S_DISTAL_CELLS = [0, 7, 14, 21, 3, 10, 17, 24, 6, 13, 20, 2, 9, 16, 23, 5, 12, 19, 1, 8, 15, 22, 4, 11, 18,
                  0, 7, 14, 21, 3, 10, 17]


def test_k_axis_and_k0_is_blind():
    """(a) The interaction-count axis is exactly {0,1,2,4,8,16,32}, k=0 = blind (no pooled summary)."""
    assert ac.K_AXIS == (0, 1, 2, 4, 8, 16, 32)
    assert ac.prefix_summary(np.random.default_rng(0).normal(size=(9, ac.FEATURE_DIM)), 0) is None
    assert ac.estimator_is_blind_at_k0() is True


def test_schedules_regenerate_and_byte_match():
    """(b) S_lift and S_distal regenerate from the frozen algorithm and byte-match the enumerated tables."""
    lift = ac.schedule_lift()
    assert [c for c, _, _ in lift] == S_LIFT_CELLS
    assert [t for _, _, t in lift] == [j % 4 for j in range(32)]
    assert lift[0] == (0, 0.12, 0) and lift[1] == (7, 0.33, 1) and lift[2] == (14, 0.54, 2) and lift[3] == (4, 0.24, 3)
    distal = ac.schedule_distal()
    assert [c for c, _, _ in distal] == S_DISTAL_CELLS
    assert distal[4] == (3, 0.18, 0) and distal[7] == (24, 0.60, 3)
    # every lift cell maps onto the 17-cell grid; distal onto the 25-cell grid
    assert all(0 <= c < 17 for c, _, _ in lift) and all(0 <= c < 25 for c, _, _ in distal)


def test_lift_grid_is_committed_17_cell():
    """(c) The lift predicted-curve length == 17 and its ell_grid matches the committed addendum grid."""
    assert len(ac.LIFT_ELLS) == 17 and ac.LIFT_ELLS[0] == 0.12 and ac.LIFT_ELLS[-1] == 0.60
    land = json.loads((MAN / "addendum_landscape.json").read_text())
    settings = land["settings"] if isinstance(land, dict) and "settings" in land else land
    grid = settings[0]["ell_grid"] if isinstance(settings, list) else land["ell_grid"]
    assert [round(x, 2) for x in grid] == list(ac.LIFT_ELLS)
    assert len(ac.DISTAL_ELLS) == 25 and ac.DISTAL_ELLS[-1] == 0.60


def test_prefix_estimator_interaction_count_order_invariant():
    """(d) The k-prefix estimator is interaction-count, order-invariant mean-pool, exactly first k."""
    rng = np.random.default_rng(1)
    feats = rng.normal(size=(10, ac.FEATURE_DIM))
    for k in (1, 2, 4, 8):
        s = ac.prefix_summary(feats, k)
        assert np.allclose(s, feats[:k].mean(axis=0))              # uses EXACTLY the first k
        perm = feats.copy(); perm[:k] = feats[:k][::-1]            # reorder within the prefix
        assert np.allclose(ac.prefix_summary(perm, k), s)          # order-invariant
    with pytest.raises(ValueError):
        ac.prefix_summary(feats[:3], 8)                            # cannot use more than available


def test_seed_banks_exact_and_disjoint():
    """(e) NEW exact-integer seed banks, disjoint from the union of ALL prior banks."""
    assert ac.SEED_BANKS["selection"] == (2300, 2301, 2302)
    assert ac.SEED_BANKS["evaluation"] == (3300, 3301, 3302, 3303, 3304)
    assert ac.SEED_BANKS["training"] == (3500, 3510, 3520)
    assert ac.seeds_disjoint() is True
    # training must not collide with the prior 3403/3413/3423
    assert set(ac.SEED_BANKS["training"]).isdisjoint({3403, 3413, 3423})


def test_metric_formulas_on_synthetic_fixtures():
    """(f) map RMSE, tau=0.5 boundary-index error, and selection regret on synthetic curves."""
    pred = np.array([0.1, 0.4, 0.6, 0.9]); meas = np.array([0.2, 0.4, 0.6, 0.8])
    assert ac.map_rmse(pred, meas) == pytest.approx(np.sqrt(((pred - meas) ** 2).mean()))
    ells = np.array([0.12, 0.15, 0.18, 0.21])
    # a monotone-rising curve crosses tau=0.5 between cells; the two curves cross at the same place -> 0 error
    assert ac.boundary_index_error(pred, pred, ells) == pytest.approx(0.0)
    assert ac.boundary_index_error(pred, meas, ells) is not None
    # selection regret: oracle argmax at cell 3 (0.9); a student picking cell 0 loses oracle[3]-oracle[0]
    oracle = np.array([0.2, 0.4, 0.6, 0.9]); student_pred = np.array([0.9, 0.1, 0.1, 0.1])
    assert ac.selection_regret(student_pred, oracle) == pytest.approx(0.9 - 0.2)
    assert ac.selection_regret(oracle, oracle) == pytest.approx(0.0)             # picks the oracle optimum
    agg = ac.aggregate_unique_groups([0.1, 0.3, None, 0.5])
    assert agg["n"] == 3 and agg["mean"] == pytest.approx(0.3)


def test_leak_free_split_partitions_15_unique_groups():
    """(g) TRAIN/VAL/TEST partition the 15 in-regime (B,w) groups; TEST never overlaps TRAIN/VAL."""
    train, val, test = set(ac.TRAIN_GROUPS), set(ac.VAL_GROUPS), set(ac.TEST_GROUPS)
    assert len(train) == 7 and len(val) == 2 and len(test) == 6
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert len(train | val | test) == 15
    assert {"R0", "R1", "R2"}.issubset(test)              # ratio pairs are A-15 invariance controls in TEST
