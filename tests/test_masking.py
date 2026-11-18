"""Phase 10: training states (trajectory and real) are built correctly."""
import numpy as np
import pytest
import torch

import distill  # noqa: F401
from distill.states import PASSES, WINDOW, RealStates, TrajectoryStates, _left
from mdlm.tokens import BOS_ID, MASK_ID


def fake_trajs(n=6, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(4, 8000, (n, WINDOW)).astype(np.int16)
    ids[:, 0] = BOS_ID
    plen = rng.integers(3, 40, n).astype(np.int16)
    step = np.zeros((n, WINDOW), np.int8)
    for i in range(n):
        step[i, plen[i]:] = rng.integers(1, PASSES + 1, WINDOW - plen[i])
    return {"ids": ids, "step": step, "conf": np.ones((n, WINDOW), np.float16), "plen": plen}


@pytest.mark.parametrize("S", [4, 8, 16])
def test_trajectory_state_partition(S):
    tr = fake_trajs()
    ts = TrajectoryStates(tr, (S,))
    for j in range(S):
        st = ts.sample(6, np.random.default_rng(j), idx=np.arange(6), S=np.full(6, S), j=np.full(6, j))
        a, b = PASSES * j // S, PASSES * (j + 1) // S
        step = torch.from_numpy(tr["step"].astype(np.int64))
        assert torch.equal(st["masked"], step > a)
        assert torch.equal(st["span"], (step > a) & (step <= b))
        assert torch.equal(st["rest"], step > b)
        assert torch.equal(st["span"] | st["rest"], st["masked"]) and not (st["span"] & st["rest"]).any()
        # state shows exactly the teacher tokens committed so far, the later state shows more
        assert torch.equal(st["x"][~st["masked"]], st["ids"][~st["masked"]])
        assert (st["x"][st["masked"]] == MASK_ID).all()
        assert ((st["x_later"] == MASK_ID) <= (st["x"] == MASK_ID)).all()
        # prompt (step 0) never masked
        for i, p in enumerate(tr["plen"]):
            assert not st["masked"][i, :p].any()


def test_trajectory_last_pass_has_no_rest():
    st = TrajectoryStates(fake_trajs(), (8,)).sample(6, np.random.default_rng(0), idx=np.arange(6),
                                                    S=np.full(6, 8), j=np.full(6, 7))
    assert not st["rest"].any() and torch.equal(st["span"], st["masked"])
    assert (st["x_later"] != MASK_ID).all()


def test_trajectory_first_pass_masks_whole_continuation():
    tr = fake_trajs()
    st = TrajectoryStates(tr, (4,)).sample(6, np.random.default_rng(0), idx=np.arange(6), S=np.full(6, 4),
                                           j=np.zeros(6, int))
    for i, p in enumerate(tr["plen"]):
        assert st["masked"][i, p:].all() and not st["masked"][i, :p].any()


class FakeWindows:
    def __init__(self, n=20):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randint(4, 8000, (n, WINDOW), generator=g)
        self.x[:, 0] = BOS_ID

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i]


def test_real_states_quota_and_prompt():
    rs = RealStates("train", np.array([5, 14, 40]), (4, 8, 16), windows=FakeWindows())
    for seed in range(5):
        st = rs.sample(16, np.random.default_rng(seed))
        n0 = WINDOW - st["plen"]
        assert (st["masked"].sum(1).numpy() == _left(n0, st["j"], st["S"])).all()
        for i, p in enumerate(st["plen"]):
            assert not st["masked"][i, :p].any()
        assert (st["x"][st["masked"]] == MASK_ID).all()
        assert torch.equal(st["x"][~st["masked"]], st["x0"][~st["masked"]])


def test_quota_matches_decoder():
    """The masks left after j of S passes equal the decoder's linear quota (pardec.decode._counts)."""
    from pardec.decode import Decoder, _counts
    for n0 in (17, 100, 251):
        for S in (4, 8, 16):
            left = n0
            for s in range(1, S + 1):
                cand = torch.zeros(1, 300, dtype=torch.bool)
                cand[0, :left] = True
                k = _counts(cand, None, torch.tensor([n0]), s, S, Decoder(steps=S))
                left -= int(k)
                assert left == int(_left(np.array(n0), np.array(s), np.array(S)))


def test_same_seed_same_states():
    tr = fake_trajs()
    a = TrajectoryStates(tr).sample(8, np.random.default_rng(3))
    b = TrajectoryStates(tr).sample(8, np.random.default_rng(3))
    assert torch.equal(a["x"], b["x"]) and torch.equal(a["span"], b["span"])
