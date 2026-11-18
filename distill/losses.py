"""Phase 7: distillation loss.

For masked positions M of a state:
    L_KD = mean_{i in M} target loss of position i
           soft target (teacher distribution q_i):  T^2 * KL( softmax(z_teacher,i / T) || softmax(z_student,i / T) )
           hard target (a token the teacher fixed): CE(z_student,i, teacher token)     (temperature does not apply)
    L_CE = mean_{i in M} CE(z_student,i, x_clean,i)                                       (real states only)
    L    = lambda_KD * L_KD + lambda_CE * L_CE,   lambda_CE = 1 - lambda_KD

KL direction: KL(teacher || student) = sum_v q_v (log q_v - log p_v), "forward" KL (the student must cover what the
teacher thinks is possible); gradients reach the student only. The teacher's [MASK] and <PAD> logits are set to
-1e4 in both distributions (the decoder never draws them; finite so that 0 * log 0 is 0, not NaN). Means are taken over the valid masked
positions only (padding never occurs in these windows; positions that are not masked never enter the loss).
"""
import torch
import torch.nn.functional as F

from mdlm.tokens import MASK_ID, PAD_ID

NEVER = -1e4


def clean_teacher(z: torch.Tensor) -> torch.Tensor:
    z = z.clone()
    if z.shape[-1] > MASK_ID:              # full vocabulary (small synthetic vocabularies in tests are left alone)
        z[..., MASK_ID] = NEVER
        z[..., PAD_ID] = NEVER
    return z


def kd_soft(z_student: torch.Tensor, z_teacher: torch.Tensor, T: float) -> torch.Tensor:
    """Per-position T^2 * KL(softmax(z_t/T) || softmax(z_s/T)); inputs (N, V) -> (N,)."""
    lt = F.log_softmax(clean_teacher(z_teacher).float() / T, dim=-1)
    ls = F.log_softmax(clean_teacher(z_student).float() / T, dim=-1)   # both over the tokens the decoder can draw
    return (T * T) * (lt.exp() * (lt - ls)).sum(-1)


def ce(z_student: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-position cross-entropy; (N, V), (N,) -> (N,)."""
    return F.cross_entropy(z_student.float(), y, reduction="none")


def total(l_kd: torch.Tensor | None, l_ce: torch.Tensor | None, lambda_kd: float) -> torch.Tensor:
    lam_ce = 1.0 - lambda_kd
    out = 0.0
    if lambda_kd > 0:
        out = out + lambda_kd * l_kd
    if lam_ce > 0:
        out = out + lam_ce * l_ce
    return out
