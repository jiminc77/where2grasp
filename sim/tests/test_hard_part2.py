"""C3 tests: Part-2 schema, split disjointness, feature dims, temporal cadence, pooling.

These run without Genesis or generated data (pure schema/logic), so they gate the
frozen s34 estimand before any histories are produced.
"""
from __future__ import annotations
import numpy as np
from sim import s34_design as sd
from sim.identify import pool_temporal, channels
from sim.history import shape_frame, temporal_shape, supported_wrench, FRAME_STEPS, M


def test_split_disjoint_complete_and_pairs_in_test():
    assert set(sd.TRAIN) | set(sd.VAL) | set(sd.TEST) == set(sd.UNIVERSE)
    assert not (set(sd.TRAIN) & set(sd.VAL)); assert not (set(sd.TRAIN) & set(sd.TEST)); assert not (set(sd.VAL) & set(sd.TEST))
    assert len(sd.TRAIN) == 7 and len(sd.VAL) == 2 and len(sd.TEST) == 6 and len(sd.UNIVERSE) == 15
    # every complete ratio pair co-located in TEST, none in TRAIN/VAL
    for ref, r in sd.FINAL_TEST_PAIRS:
        assert ref in sd.TEST and r in sd.TEST
        assert ref not in sd.TRAIN + sd.VAL and r not in sd.TRAIN + sd.VAL


def test_grouped_cv_folds_train_only():
    f = sd.folds(sd.TRAIN, 3)
    flat = [x for v in f.values() for x in v]
    assert sorted(flat) == sorted(sd.TRAIN)              # every TRAIN setting in exactly one fold
    assert all(x not in sd.TEST + sd.VAL for x in flat)  # no TEST/VAL leakage into folds


def test_temporal_cadence_is_seven_frames():
    assert FRAME_STEPS == [60, 120, 180, 240, 300, 360]  # + settled = 7 frames
    assert M == 8


def test_feature_dims():
    # per-frame 16 (M=8 (y,z)); task temporal 112 (7x16); probe 16; probe-enriched 128 (8x16)
    v = np.zeros((20, 3)); v[:, 0] = np.linspace(0, 0.18, 20); v[2:, 2] = -0.001 * np.arange(18) ** 2
    assert shape_frame(v).shape == (16,)
    assert temporal_shape([v] * 7).shape == (112,)
    assert pool_temporal(np.arange(112.0)).shape == (32,)          # mean + last frame -> 32
    # channel feature vectors on a synthetic history record
    class HZ(dict):
        def __getitem__(self, k):
            return {'proprio': np.zeros((1, 8)), 'shape': np.zeros((1, 112)),
                    'probe_shape': np.zeros((1, 16)), 'wrench': np.zeros(1)}[k]
    ch = channels(HZ(), 0)
    assert ch['proprioception-only'].shape == (8,)
    assert ch['+shape'].shape == (8 + 32,)               # pooled task
    assert ch['+wrench'].shape == (8 + 32 + 1,)
    assert ch['probe+shape'].shape == (8 + 16 + 32,)     # probe + pooled task
    assert ch['probe+shape+wrench'].shape == (8 + 16 + 32 + 1,)


def test_supported_wrench_positive_and_scales():
    w1 = supported_wrench(0.001, 14); w2 = supported_wrench(0.002, 14)
    assert w1 > 0 and abs(w2 / w1 - 2.0) < 1e-9         # linear in segment mass
