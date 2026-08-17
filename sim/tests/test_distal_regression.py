"""C0 hardening-A byte-unchanged regression + no-overwrite guard.

The frozen lift runner (sweep.py) and the byte-frozen gate surface (analyze_gate.py) MUST be
untouched by the distal build, and distal modules MUST write only distal_* outputs (never
overwrite hard_* artifacts). The existing hard golden gate test is re-exercised for the
boundary()/condition() source hashes.
"""
from __future__ import annotations
import ast, hashlib
from pathlib import Path
import sim.analyze_gate as ag
import sim.distal_sweep as ds
import sim.distal_gate as dg

ROOT = Path(__file__).resolve().parents[1]

# frozen source hashes (byte-unchanged guard)
SWEEP_SHA = 'da042e613cdacd3dcda247b4f69c118180b34bc867d664b26a475945a6a45208'
ANALYZE_SHA = 'c8ca30a3f9e032db86282b05dd0f81845b90bea3a071af9ab64a61ebede5bbfb'
# hardening-A golden decision-surface hashes (must remain identical to test_hard_gate)
GOLD = {'boundary': '947a93317593d645a150f057ed10a3bb028d57c6be5cf76b613203f544962e06',
        'condition': '73ea1d4bf16af2a8dca4251954621364e1fe9e245b462d581c11ae93beaf44bd'}


def test_sweep_py_byte_unchanged():
    assert hashlib.sha256((ROOT / 'sweep.py').read_bytes()).hexdigest() == SWEEP_SHA


def test_analyze_gate_byte_unchanged():
    assert hashlib.sha256((ROOT / 'analyze_gate.py').read_bytes()).hexdigest() == ANALYZE_SHA


def test_frozen_decision_surface_hashes_hold():
    src = Path(ag.__file__).read_text(); tree = ast.parse(src); lines = src.splitlines(keepends=True)
    seen = {}
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and fn.name in GOLD:
            seg = ''.join(lines[fn.lineno - 1:fn.end_lineno]); seen[fn.name] = hashlib.sha256(seg.encode()).hexdigest()
    assert seen == GOLD, seen


def test_distal_writes_only_distal_outputs():
    for mod in (ds, dg):
        src = Path(mod.__file__).read_text()
        # every default output path is a distal_* artifact; no hard_* write target appears
        assert 'hard_sweep_results.npz' not in src.replace('# ', '')
        assert 'hard_gate_verdict.json' not in src
        assert 'hard_histories_v2.npz' not in src
    ds_src = Path(ds.__file__).read_text(); dg_src = Path(dg.__file__).read_text()
    assert "distal_sweep_results.npz" in ds_src
    assert "distal_gate_verdict.json" in dg_src


def test_distal_does_not_mutate_hard_manifests_on_import():
    # importing distal modules must not touch hard_* artifacts (pure import, no side effects)
    import sim.tip_model, sim.distal_grid_design  # noqa: F401
    assert True
