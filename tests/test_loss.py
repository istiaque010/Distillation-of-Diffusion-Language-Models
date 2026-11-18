"""Phase 10: loss direction, temperature, masking, teacher freeze, student update."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

import distill  # noqa: F401
from distill import losses
from distill.models import make_student
from distill.states import TrajectoryStates
from distill.train_student import logits_at, traj_loss
from mdlm.model import MaskedDiffusionLM, MDLMConfig
from mdlm.tokens import MASK_ID, PAD_ID
from tests.test_masking import fake_trajs


def tiny(seed=0, layers=2):
    torch.manual_seed(seed)
    return MaskedDiffusionLM(MDLMConfig(n_layer=layers, n_head=2, n_embd=32, dropout=0.0)).eval()


def test_kd_zero_for_identical_logits():
    z = torch.randn(7, 50)
    assert torch.allclose(losses.kd_soft(z, z, 2.0), torch.zeros(7), atol=1e-6)


def test_kd_direction_is_teacher_to_student():
    zt, zs = torch.randn(5, 30), torch.randn(5, 30)
    q, p = F.softmax(zt, -1), F.softmax(zs, -1)
    forward = (q * (q.log() - p.log())).sum(-1)      # KL(teacher || student)
    reverse = (p * (p.log() - q.log())).sum(-1)
    got = losses.kd_soft(zs, zt, 1.0)
    assert torch.allclose(got, forward, atol=1e-5) and not torch.allclose(got, reverse, atol=1e-3)


@pytest.mark.parametrize("T", [1.0, 2.0, 4.0])
def test_temperature_scaling(T):
    zt, zs = torch.randn(4, 20), torch.randn(4, 20)
    q, p = F.log_softmax(zt / T, -1), F.log_softmax(zs / T, -1)
    assert torch.allclose(losses.kd_soft(zs, zt, T), T * T * (q.exp() * (q - p)).sum(-1), atol=1e-5)


def test_teacher_mask_and_pad_never_targets():
    zt = torch.zeros(1, MASK_ID + 1)
    zt[0, MASK_ID] = 50.0
    zt[0, PAD_ID] = 50.0
    q = F.softmax(losses.clean_teacher(zt), -1)
    assert q[0, MASK_ID] == 0 and q[0, PAD_ID] == 0
    assert torch.isfinite(losses.kd_soft(torch.randn(1, MASK_ID + 1), zt, 2.0)).all()


def test_total_weights():
    a, b = torch.tensor(2.0), torch.tensor(4.0)
    assert losses.total(a, b, 0.5) == 3.0
    assert losses.total(None, b, 0.0) == 4.0 and losses.total(a, None, 1.0) == 2.0


def test_logits_only_at_requested_positions():
    m = tiny()
    x = torch.randint(4, 8000, (2, 256))
    where = torch.zeros(2, 256, dtype=torch.bool)
    where[0, 5] = where[1, 200] = True
    full = m(x)
    assert torch.allclose(logits_at(m, x, where), full[where], atol=1e-5)


def test_traj_loss_uses_only_masked_positions_and_right_targets():
    t = tiny(0)
    s = make_student(t, dropout=0.0).eval()
    st = TrajectoryStates(fake_trajs(3), (8,)).sample(3, np.random.default_rng(0), idx=np.arange(3),
                                                       S=np.full(3, 8), j=np.full(3, 2))
    per, span = traj_loss(s, t, st, 2.0, "traj")
    assert per.numel() == int(st["masked"].sum()) and span.sum() == st["span"].sum()
    # span positions: CE against the teacher's token
    z = s(st["x"])[st["masked"]][span]
    y = st["ids"][st["masked"]][span]
    assert torch.allclose(per[span], F.cross_entropy(z, y, reduction="none"), atol=1e-4)
    # same-state targets with an identical student: zero loss
    per_state, _ = traj_loss(s, t, st, 2.0, "state")
    assert per_state.abs().max() < 1e-4


def test_teacher_frozen_student_updates():
    t = tiny(0)
    for p in t.parameters():
        p.requires_grad_(False)
    before = {k: v.clone() for k, v in t.state_dict().items()}
    s = make_student(t, dropout=0.1)
    s0 = {k: v.clone() for k, v in s.state_dict().items()}
    opt = torch.optim.AdamW([p for p in s.parameters() if p.requires_grad], lr=1e-2)
    st = TrajectoryStates(fake_trajs(3), (4,)).sample(3, np.random.default_rng(1))
    per, _ = traj_loss(s, t, st, 2.0, "traj")
    per.mean().backward()
    assert all(p.grad is None for p in t.parameters())
    opt.step()
    assert all(torch.equal(v, before[k]) for k, v in t.state_dict().items())
    assert any(not torch.equal(v, s0[k]) for k, v in s.state_dict().items())
    assert s.head.weight is s.tok_emb.weight


def test_shallow_student_keeps_chosen_layers():
    t = tiny(0, layers=4)
    s = make_student(t, layers=[0, 3], dropout=0.0)
    assert len(s.blocks) == 2 and s.cfg.n_layer == 2
    assert torch.equal(s.blocks[1].mlp[0].weight, t.blocks[3].mlp[0].weight)
    assert s(torch.randint(4, 8000, (1, 256))).shape == (1, 256, MASK_ID + 1)
