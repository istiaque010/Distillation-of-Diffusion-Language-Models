"""Teacher and student models (Phases 1 and 4).

teacher  = Project 3 mdlm_base (MDLM-13.9M), loaded through Project 4's loader (sha256-checked there), frozen:
           eval mode, requires_grad False everywhere.
student  = same architecture and tokenizer, initialised as an exact copy of the teacher, or a shallower copy that
           keeps some of the teacher's layers (layers=[0, 2, 3, 5] = 4 layers). [MASK], <BOS>, <EOS> and <PAD> ids are
           shared because the tokenizer is shared.
"""
import copy
import os

import torch

import distill
from bench.load_models import load_model
from mdlm.model import MaskedDiffusionLM, MDLMConfig, count_parameters


def load_teacher():
    t = load_model("mdlm13")
    t.eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


def make_student(teacher, layers=None, dropout: float = 0.1):
    """Copy of the teacher (all layers, or only `layers`), trainable, with the given dropout."""
    s = copy.deepcopy(teacher)
    if layers is not None:
        s.blocks = torch.nn.ModuleList([s.blocks[i] for i in layers])
        s.cfg = MDLMConfig(**{**s.cfg.__dict__, "n_layer": len(layers)})
    for m in s.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = dropout
    s.cfg.dropout = dropout
    s.head.weight = s.tok_emb.weight          # keep the tied embedding / output layer tied
    for p in s.parameters():
        p.requires_grad_(True)
    return s.train()


def save_student(path, model, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), "model_config": dict(model.cfg.__dict__), "meta": meta}, path)


def load_student(run: str, device: str = "cpu"):
    ck = torch.load(os.path.join(distill.CHECKPOINTS, run, "student.pt"), map_location=device, weights_only=False)
    m = MaskedDiffusionLM(MDLMConfig(**ck["model_config"]))
    m.load_state_dict(ck["model"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_system(model_key: str):
    """'teacher' or a student run name."""
    return load_teacher() if model_key == "teacher" else load_student(model_key)


__all__ = ["load_teacher", "make_student", "save_student", "load_student", "load_system", "count_parameters"]
