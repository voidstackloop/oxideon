"""Complete optimizer and learning rate scheduler suite for Oxideon.

Includes the standard Optimizer base class, traditional optimizers (SGD, RMSprop, Adagrad,
Adadelta, Adamax, ASGD), modern adaptive optimizers (Adam, AdamW, NAdam, RAdam),
second-order quasi-Newton algorithms (L-BFGS), and frontier optimizers (Muon,
Sophia, Prodigy, Shampoo, Lookahead, RiemannianAdam, ScheduleFreeAdamW).
"""

from ..oxideon import (
    SGD,
    Adam,
    AdamW,
    Muon,
    RiemannianAdam,
    ScheduleFreeAdamW,
)
from ..optim_ext import (
    ProdigyOptimizer,
    SophiaOptimizer,
    ShampooOptimizer,
    LookaheadOptimizer,
)
from .optimizer import (
    Optimizer,
    RMSprop,
    Adagrad,
    Adadelta,
    Adamax,
    NAdam,
    RAdam,
    ASGD,
    LBFGS,
    Lion,
    Lamb,
    Adafactor,
    Rprop,
)
from . import swa_utils
from . import lr_scheduler
from .lr_scheduler import (
    StepLR,
    MultiStepLR,
    ConstantLR,
    ExponentialLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    OneCycleLR,
    ReduceLROnPlateau,
    WarmupCosineScheduler,
    PolynomialLR,
    CyclicLR,
    ChainedScheduler,
    LambdaLR,
    MultiplicativeLR,
    SequentialLR,
)

Prodigy = ProdigyOptimizer
Sophia = SophiaOptimizer
Shampoo = ShampooOptimizer
Lookahead = LookaheadOptimizer

__all__ = [
    # Base
    "Optimizer",
    # First-Order & Adaptive Optimizers
    "SGD",
    "RMSprop",
    "Adagrad",
    "Adadelta",
    "Adam",
    "AdamW",
    "Adamax",
    "NAdam",
    "RAdam",
    "ASGD",
    # Quasi-Newton & Second-Order
    "LBFGS",
    # Frontier & Specialized Optimizers
    "Lion",
    "Lamb",
    "Adafactor",
    "Rprop",
    "swa_utils",
    "Muon",
    "Prodigy",
    "ProdigyOptimizer",
    "Sophia",
    "SophiaOptimizer",
    "Shampoo",
    "ShampooOptimizer",
    "Lookahead",
    "LookaheadOptimizer",
    "RiemannianAdam",
    "ScheduleFreeAdamW",
    # Schedulers
    "lr_scheduler",
    "StepLR",
    "MultiStepLR",
    "ConstantLR",
    "ExponentialLR",
    "CosineAnnealingLR",
    "CosineAnnealingWarmRestarts",
    "LinearLR",
    "OneCycleLR",
    "ReduceLROnPlateau",
    "WarmupCosineScheduler",
    "PolynomialLR",
    "CyclicLR",
    "ChainedScheduler",
    "LambdaLR",
    "MultiplicativeLR",
    "SequentialLR",
]
