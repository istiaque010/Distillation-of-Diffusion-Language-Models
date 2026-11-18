"""Project 7: step distillation of the Project 3 masked diffusion language model (MDLM-13.9M).

Code package `distill`. Importing it makes Project 6's `pardec` importable (the decoder used for every model here),
which in turn makes Project 5's `adaptive`, Project 4's `bench` (model loader, tokenizer, prompts, scorer, quality
measures, paired bootstrap), Project 3's `mdlm` (model, masking, training helpers) and Project 1's `src` importable.
Projects 1-6 are only read, never changed; bytecode writing is off so nothing is written into them.
"""
import os
import sys

sys.dont_write_bytecode = True

PROJECT7 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(PROJECT7)
PROJECT6 = os.path.join(ROOT, "project6")
CONFIGS = os.path.join(PROJECT7, "configs")
PROMPTS = os.path.join(PROJECT7, "prompts")
CHECKPOINTS = os.path.join(PROJECT7, "checkpoints")
RESULTS = os.path.join(PROJECT7, "results")
METRICS = os.path.join(RESULTS, "metrics")
GENERATIONS = os.path.join(RESULTS, "generations")
FIGURES = os.path.join(RESULTS, "figures")
LABELS = os.path.join(RESULTS, "error_labels")
TRAJ = os.path.join(RESULTS, "trajectories")

if PROJECT6 not in sys.path:
    sys.path.insert(0, PROJECT6)
import pardec  # noqa: E402,F401  (Project 6; adds Projects 5, 4, 3 and 1 to sys.path)
import bench  # noqa: E402


def save_json(obj, *path):
    return bench.save_json(obj, *path)


def load_json(*path):
    return bench.load_json(*path)
