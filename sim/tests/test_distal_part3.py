"""C0 Part-3 structural tests (no Genesis / no generated data): split hygiene, transfer-probe
leak-free construction, action metadata, and the frozen-encoder contract.
"""
from __future__ import annotations
import numpy as np
from sim import distal_s34_design as sd
from sim.identify import pool_temporal


def test_split_disjoint_complete_and_pairs_in_test():
    assert set(sd.TRAIN) | set(sd.VAL) | set(sd.TEST) == set(sd.UNIVERSE)
    assert not (set(sd.TRAIN) & set(sd.VAL)); assert not (set(sd.TRAIN) & set(sd.TEST)); assert not (set(sd.VAL) & set(sd.TEST))
    assert len(sd.TRAIN) == 7 and len(sd.VAL) == 2 and len(sd.TEST) == 6 and len(sd.UNIVERSE) == 15
    for ref, r in sd.FINAL_TEST_PAIRS:
        assert ref in sd.TEST and r in sd.TEST and ref not in sd.TRAIN + sd.VAL and r not in sd.TRAIN + sd.VAL


def test_transfer_split_leak_free():
    # distal TEST must be disjoint from the SOURCE encoder's TRAIN+VAL (leak-free transfer probe)
    assert not (set(sd.TEST) & (set(sd.SOURCE_TRAIN) | set(sd.SOURCE_VAL)))
    # and the distal head fits only on distal TRAIN/VAL, disjoint from TEST
    assert not ((set(sd.TRAIN) | set(sd.VAL)) & set(sd.TEST))


def test_grouped_cv_folds_train_only():
    f = sd.folds(sd.TRAIN, 3)
    flat = [x for v in f.values() for x in v]
    assert sorted(flat) == sorted(sd.TRAIN)
    assert all(x not in sd.TEST + sd.VAL for x in flat)


def test_pooled_dim_and_action_metadata_contract():
    assert pool_temporal(np.arange(112.0)).shape == (32,)          # task shape pooled to 32
    # the student input includes action metadata (A-17): 8 + 32 + 2 = 42; source encoder is 40 (no action)
    # (validated against the frozen manifest spec below)


def test_manifest_spec_fields(tmp_path):
    # build to a temp file (no C2 inputs needed; input shas may be null pre-data)
    out = sd.build()  # writes distal_s34_manifest.json; returns sha
    import json
    from pathlib import Path
    m = json.loads((Path(sd.MAN) / 'distal_s34_manifest.json').read_text())
    assert m['history_policy']['action_metadata'] is True
    assert m['features']['action_dim'] == 2 and '42-D' in m['features']['student_input']
    assert m['critic_head']['kind'] == 'J_regression'
    assert m['primary']['name'] == 'selection_regret' and m['primary']['J_inf'] == -1.0
    assert m['transfer']['non_inferiority_margin'] == 0.05
    assert 'invariance controls' in m['primary']['ratio_pairs']
    assert m['multi_seed_note'] and len(m['training_seeds']) >= 2
    assert m['honest_nulls']['Q1'] and m['honest_nulls']['Q2'] and m['honest_nulls']['Q3']
    # clean up the generated manifest so C0 stays code-only
    (Path(sd.MAN) / 'distal_s34_manifest.json').unlink()
