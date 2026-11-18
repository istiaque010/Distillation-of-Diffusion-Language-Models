"""Phase 6: training states and targets.

A student with S passes (S in {4, 8, 16}) sees, before its pass j+1 (j = 0..S-1), a window in which a share j/S of the
continuation is fixed and the rest is [MASK]. Two sources of such states, used side by side in every batch:

TRAJECTORY STATES (teacher supervision). Take one stored teacher trajectory (32 passes) and a student pass j. One
student pass covers 32/S teacher passes, so
    a = 32 j / S,  b = 32 (j + 1) / S
    state      x_a = teacher tokens committed at passes <= a, [MASK] elsewhere      (student input)
    span       positions the teacher committed in passes a+1 .. b                   (what ONE student pass must fix)
    rest       positions still masked after teacher pass b
    later      x_b = the teacher's state after pass b
These are exactly the states the student meets when it decodes with the same rule (random order + no-adjacent,
linear quota), only with the teacher's tokens instead of its own.

REAL STATES (clean labels). A real training window (story-aligned, as in Project 3), a prompt length drawn from the
training prompts, and a random share 1 - j/S of the continuation masked (same quota as the decoder).

All randomness comes from one numpy Generator per run, seeded in the config, so every run sees the same states.
"""
import numpy as np
import torch

from mdlm.data import StoryWindows
from mdlm.tokens import MASK_ID

PASSES = 32
WINDOW = 256


def _left(n0: np.ndarray, j: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Masks left after j of S passes with the decoder's linear quota: floor(n0 (1 - j / S))."""
    return np.floor(n0 * (1.0 - j / S)).astype(np.int64)


class TrajectoryStates:
    def __init__(self, traj: dict, budgets=(4, 8, 16)):
        self.ids = traj["ids"].astype(np.int64)
        self.step = traj["step"].astype(np.int64)
        self.plen = traj["plen"].astype(np.int64)
        self.budgets = np.array(budgets)
        assert all(PASSES % s == 0 for s in budgets)

    def sample(self, n: int, rng: np.random.Generator, idx=None, S=None, j=None) -> dict:
        idx = rng.integers(0, len(self.ids), n) if idx is None else np.asarray(idx)
        S = rng.choice(self.budgets, n) if S is None else np.asarray(S)
        j = rng.integers(0, S) if j is None else np.asarray(j)
        a, b = PASSES * j // S, PASSES * (j + 1) // S
        ids, step = self.ids[idx], self.step[idx]
        known = step <= a[:, None]
        later_known = step <= b[:, None]
        x = np.where(known, ids, MASK_ID)
        x_later = np.where(later_known, ids, MASK_ID)
        return {"x": torch.from_numpy(x), "x_later": torch.from_numpy(x_later), "ids": torch.from_numpy(ids),
                "masked": torch.from_numpy(~known), "span": torch.from_numpy(later_known & ~known),
                "rest": torch.from_numpy(~later_known), "S": S, "j": j}


class RealStates:
    def __init__(self, split: str, plen_pool: np.ndarray, budgets=(4, 8, 16), windows: StoryWindows | None = None):
        self.ds = windows or StoryWindows(split, WINDOW)
        self.plen_pool = np.asarray(plen_pool, dtype=np.int64)
        self.budgets = np.array(budgets)

    def sample(self, n: int, rng: np.random.Generator) -> dict:
        w = rng.integers(0, len(self.ds), n)
        x0 = torch.stack([self.ds[int(i)] for i in w])
        plen = rng.choice(self.plen_pool, n)
        S = rng.choice(self.budgets, n)
        j = rng.integers(0, S)
        n0 = WINDOW - plen
        left = _left(n0, j, S)
        u = rng.random((n, WINDOW))
        u[np.arange(WINDOW)[None, :] < plen[:, None]] = 2.0          # prompt positions are never masked
        rank = np.argsort(np.argsort(u, axis=1), axis=1)              # rank of each position by u
        masked = rank < left[:, None]
        masked_t = torch.from_numpy(masked)
        x = torch.where(masked_t, torch.full_like(x0, MASK_ID), x0)
        return {"x": x, "x0": x0, "masked": masked_t, "S": S, "j": j, "plen": plen}
