"""PhomiNeuro — a neural-field surrogate for time-resolved near-infrared fluence in the head.

    from phomineuro import Scene, Predictor, ROOT
    s = Scene("scb16", "Cz", ROOT, dev, cache_dir=".../pyr", vista3d_ckpt=..., weights_dir=...)
    p = Predictor(".../phomineuro_s0.pt", dev)
    logphi = p.at_points(s, [[120, 130, 170]])          # (1,10) log10 fluence, all gates

Predictions are log10 fluence at 810 nm on a 1 mm MNI grid, in the units of the Monte-Carlo
reference they were trained against.
"""
import os

__version__ = "0.1.0"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "weights")

from .scene import Scene                                             # noqa: E402
from .predict import Predictor, load_ensemble, HERO3                 # noqa: E402
from .features import GATE_NS, N_STEP, T_ENC, t_code                 # noqa: E402

__all__ = ["Scene", "Predictor", "load_ensemble", "HERO3", "ROOT", "WEIGHTS",
           "GATE_NS", "N_STEP", "T_ENC", "t_code"]
