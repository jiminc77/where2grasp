"""Item 1 (simulation wrap-up): the k-INTERACTION adaptation curve (adjudication C-2 / finding 32).

The interaction-count axis: how does performance improve with the NUMBER OF INTERACTIONS k, from a
cold start (k=0 = the property-blind prior), read against the privileged-teacher ceiling. This is
DISTINCT from the already-reported within-interaction frame-truncation axis (discipline A-22): the
frame-truncation figure is the history-variant comparison; "adaptation curve" is reserved here for
interaction-prefix curves.

Acquisition rule = a FIXED deterministic schedule (information-sufficiency vs k, NOT the deployment
protocol). The k-prefix estimator MEAN-POOLS the first k interactions' committed 42-D addendum
features into a fixed order-invariant summary fed to the frozen head MLP[32,32]; k=0 bypasses the
pooled summary -> the property-blind prior head (identical to the blind baseline). PRIMARY task =
lift-and-clear on the COMMITTED 17-cell / 0.03 grid; SECONDARY = distal spanning on the 25-cell /
0.02 grid (a DISTINCT estimand; endpoints non-comparable across grids).

This module holds the frozen, fully-determined estimand core (schedules, k-axis, mean-pool estimator,
metric formulas, A-15 aggregation, seed banks). The GPU k-sweep (C2) reuses the committed addendum /
spanning training pipeline; nothing here forks those cores.
"""
from __future__ import annotations

import numpy as np

# --- interaction-count axis (frozen) ---
K_AXIS = (0, 1, 2, 4, 8, 16, 32)
N_SCHEDULE_STEPS = 32                    # enough interactions to reach k_max = 32

# --- PRIMARY lift-and-clear grid (COMMITTED 17-cell / 0.03) ---
LIFT_ELL_LO = 0.12
LIFT_ELL_STEP = 0.03
LIFT_N_CELLS = 17
LIFT_ELLS = tuple(round(LIFT_ELL_LO + LIFT_ELL_STEP * i, 2) for i in range(LIFT_N_CELLS))

# --- SECONDARY distal spanning grid (25-cell / 0.02) ---
DISTAL_ELL_LO = 0.12
DISTAL_ELL_STEP = 0.02
DISTAL_N_CELLS = 25
DISTAL_ELLS = tuple(round(DISTAL_ELL_LO + DISTAL_ELL_STEP * i, 2) for i in range(DISTAL_N_CELLS))

N_TEMPLATES = 4                          # 0=linear, 1=ease, 2=arc+0.025, 3=arc-0.025

# --- addendum leak-free split (addendum_manifest.json:splits) ---
TRAIN_GROUPS = ("B1_w0", "B1_w2", "B2_w0", "B3_w0", "B3_w1", "B4_w0", "B4_w2")
VAL_GROUPS = ("B2_w2", "B4_w1")
TEST_GROUPS = ("B1_w1", "R0", "B3_w2", "R1", "B2_w1", "R2")

# --- NEW exact-integer seed banks (frozen; disjoint from ALL prior banks) ---
SEED_BANKS = {"selection": (2300, 2301, 2302), "evaluation": (3300, 3301, 3302, 3303, 3304),
              "history": (2500, 2501), "training": (3500, 3510, 3520)}
PRIOR_SEED_UNION = frozenset(
    # selection
    {2000, 2001, 2002, 2100, 2101, 2102, 2200, 2201, 2202}
    # evaluation
    | {3000, 3001, 3002, 3003, 3004, 3100, 3101, 3102, 3103, 3104, 3200, 3201, 3202, 3203, 3204}
    # history (1000-series + 2100/2200 pairs)
    | {1000, 1001, 1002, 2100, 2101, 2200, 2201}
    # training
    | {3403, 3413, 3423})

FEATURE_DIM = 42                         # proprio(8) + pooled_temporal_shape(112->32) + action(2)


# --------------------------------------------------------------------------- #
# fixed acquisition schedule (fully enumerated / regeneratable)
# --------------------------------------------------------------------------- #
def acquisition_schedule(n_cells, ell_step, n_steps=N_SCHEDULE_STEPS, ell_lo=0.12):
    """Frozen deterministic schedule S: the (k+1)-th interaction (0-indexed step j = k) uses
    cell(j) = (7*j) mod n_cells (stride 7 coprime to 17 and 25 -> a full permutation over the grid),
    template(j) = j mod 4, ell(j) = ell_lo + ell_step*cell(j). Returns a list of (cell, ell, template)."""
    out = []
    for j in range(n_steps):
        cell = (7 * j) % n_cells
        out.append((cell, round(ell_lo + ell_step * cell, 2), j % N_TEMPLATES))
    return out


def schedule_lift(n_steps=N_SCHEDULE_STEPS):
    return acquisition_schedule(LIFT_N_CELLS, LIFT_ELL_STEP, n_steps, LIFT_ELL_LO)


def schedule_distal(n_steps=N_SCHEDULE_STEPS):
    return acquisition_schedule(DISTAL_N_CELLS, DISTAL_ELL_STEP, n_steps, DISTAL_ELL_LO)


# --------------------------------------------------------------------------- #
# k-prefix estimator (mean-pool the first k interactions; k=0 == blind)
# --------------------------------------------------------------------------- #
def prefix_summary(features, k):
    """Order-invariant MEAN-POOL of the first k interactions' 42-D feature vectors into a fixed 42-D
    summary. k=0 returns None -> the caller uses the property-blind prior head (identical to blind).
    `features` is a (>=k, 42) array of per-interaction committed addendum features."""
    if k <= 0:
        return None                       # k=0 bypasses the pooled summary -> property-blind prior
    f = np.asarray(features, dtype=float)
    if f.shape[0] < k:
        raise ValueError(f"need >= {k} interaction features, got {f.shape[0]}")
    return f[:k].mean(axis=0)             # mean-pool (order-invariant; uses EXACTLY the first k)


def estimator_is_blind_at_k0():
    """k=0 uses no interaction information -> byte-identical to the blind baseline."""
    return prefix_summary(np.random.default_rng(0).normal(size=(5, FEATURE_DIM)), 0) is None


# --------------------------------------------------------------------------- #
# metric formulas (frozen; per k, aggregated over unique-(B,w) TEST groups, A-15)
# --------------------------------------------------------------------------- #
def map_rmse(pred_curve, measured_curve):
    """Held-out map RMSE = sqrt(mean over grasp cells of (pred - measured)^2)."""
    p = np.asarray(pred_curve, float); m = np.asarray(measured_curve, float)
    return float(np.sqrt(np.mean((p - m) ** 2)))


def tau_crossing_cell(curve, ells, tau=0.5):
    """Grid-cell index of the tau=0.5 boundary via linear interpolation of the success curve."""
    y = np.asarray(curve, float); g = np.asarray(ells, float)
    step = g[1] - g[0]
    for i in range(len(y) - 1):
        if (y[i] - tau) * (y[i + 1] - tau) <= 0 and y[i] != y[i + 1]:
            t = (tau - y[i]) / (y[i + 1] - y[i])
            ell_cross = g[i] + t * (g[i + 1] - g[i])
            return (ell_cross - g[0]) / step
    return None


def boundary_index_error(pred_curve, measured_curve, ells, tau=0.5):
    """tau=0.5 boundary-index error in grid cells: |cell(pred crossing) - cell(measured crossing)|."""
    cp = tau_crossing_cell(pred_curve, ells, tau); cm = tau_crossing_cell(measured_curve, ells, tau)
    if cp is None or cm is None:
        return None
    return float(abs(cp - cm))


def selection_regret(pred_curve, oracle_curve):
    """Selection regret = J(oracle argmax grasp) - J(student argmax grasp), both scored on the oracle
    (measured) curve. >= 0; 0 iff the student picks an oracle-optimal grasp."""
    o = np.asarray(oracle_curve, float); p = np.asarray(pred_curve, float)
    return float(o[int(np.argmax(o))] - o[int(np.argmax(p))])


def aggregate_unique_groups(per_group_values):
    """A-15: aggregate a per-unique-(B,w)-group metric to a mean + std at the GROUP level (not
    per-rollout). Ratio pairs (R0/R1/R2) are ordinary groups here and serve as invariance controls."""
    v = np.asarray([x for x in per_group_values if x is not None], float)
    if v.size == 0:
        return dict(mean=float("nan"), std=float("nan"), n=0)
    return dict(mean=float(v.mean()), std=float(v.std()), n=int(v.size))


# --------------------------------------------------------------------------- #
# seed-bank disjointness (A-16; NEW banks vs the union of ALL prior banks)
# --------------------------------------------------------------------------- #
def seeds_disjoint():
    """Every NEW seed bank is disjoint from the union of all prior selection/evaluation/history/
    training banks."""
    allnew = set().union(*(set(v) for v in SEED_BANKS.values()))
    return allnew.isdisjoint(PRIOR_SEED_UNION)
