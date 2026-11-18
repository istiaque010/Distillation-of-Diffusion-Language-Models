"""Phase 10: low-step sampling with a student (Project 6's decoder) and the denoiser's attention."""
import pytest
import torch

import distill  # noqa: F401
from adaptive.generate import SAMPLING
from distill.models import make_student
from distill.run import split_name, system
from mdlm.tokens import BOS_ID, MASK_ID
from pardec.decode import decode_ids, parse
from tests.test_loss import tiny

PROMPTS = [[10, 11, 12], [20, 21, 22, 23, 24, 25, 26], [30]]


def student():
    return make_student(tiny(1), dropout=0.0).eval()


def test_no_causal_mask():
    """Changing a LATER token changes the prediction at an EARLIER position (bidirectional denoiser)."""
    m = student()
    x = torch.randint(4, 8000, (1, 256))
    y = x.clone()
    y[0, 200] = (y[0, 200] + 1) % 8000
    assert not torch.allclose(m(x)[0, 10], m(y)[0, 10])


@pytest.mark.parametrize("S", [4, 8, 16])
def test_student_completes_at_every_budget(S):
    m = student()
    d = parse(f"full-random-K{S}+noadj+lazy")
    x, info = decode_ids(m, PROMPTS, 64, d, SAMPLING["topk20"], torch.Generator().manual_seed(0), trace=True)
    assert info["nfe"] <= S and (x != MASK_ID).all()
    for b, p in enumerate(PROMPTS):
        assert x[b, 0] == BOS_ID and x[b, 1:1 + len(p)].tolist() == p          # prompt preserved
    masks = [int((s == MASK_ID).sum()) for s in info["states"]]
    assert all(a > b for a, b in zip(masks, masks[1:]))                       # monotone decrease
    assert masks[-1] == 0


def test_never_unmasks_more_than_remain():
    m = student()
    d = parse("full-random-K4+noadj")
    x, info = decode_ids(m, PROMPTS, 40, d, SAMPLING["topk20"], torch.Generator().manual_seed(2), trace=True)
    for a, b in zip(info["states"], info["states"][1:]):
        newly = (a == MASK_ID) & (b != MASK_ID)
        assert int(newly.sum()) <= int((a == MASK_ID).sum())
        assert ((a != MASK_ID) <= (b != MASK_ID)).all() and torch.equal(a[a != MASK_ID], b[a != MASK_ID])


def test_same_seed_same_output():
    m = student()
    d = parse("full-random-K8+noadj+stop+lazy")
    outs = [decode_ids(m, PROMPTS, 64, d, SAMPLING["topk20"], torch.Generator().manual_seed(5))[0] for _ in range(2)]
    assert torch.equal(outs[0], outs[1])


def test_early_stop_and_eos_handling():
    m = student()
    d = parse("full-random-K8+noadj+stop+lazy")
    x, _ = decode_ids(m, PROMPTS, 64, d, SAMPLING["topk20"], torch.Generator().manual_seed(0))
    assert x.shape == (3, 64)


def test_system_names():
    nm = system("kd", 8)
    assert nm == "kd@full-random-K8+noadj+stop+lazy"
    assert split_name(nm) == ("kd", "full-random-K8+noadj+stop+lazy", "topk20")
    assert split_name(system("teacher", 8, sampling="topk20t0.8"))[2] == "topk20t0.8"
    assert system("ce", 4, conf=True) == "ce@full-conf-K4+stop+lazy"
    assert "topk20t0.8" in SAMPLING and SAMPLING["topk20t0.8"]["temperature"] == 0.8
