from dataclasses import dataclass, field
from typing import Final, Dict, List, TypeVar, Generic
from enum import Enum, auto


class STgt(Enum):
    UP = auto()
    DWN = auto()
    CONS = auto()

    def __str__(self):
        return self.name.replace('_', ' ').title()


class K:
    # INDICATORS: Final[bool] = False  # Use technical indicators
    INDICATORS: Final[bool] = True  # Use technical indicators
    TARGET: Final[STgt] = STgt.CONS  # Target
    PCT: Final[float] = 0.03  # 3%

    # Dirs/Paths
    TRY_DIR: Final[str] = './ml/.try{}_{}_{}/'.format(
        "" if not INDICATORS else "_indicators", TARGET.name.lower(), PCT)
    DATA_DIR: Final[str] = f'./ml/{TRY_DIR}/data/'
    MDL_DIR: Final[str] = f'./ml/{TRY_DIR}/best_model.pth'
    RPRT_DIR: Final[str] = f'./ml/{TRY_DIR}/report.html.pth'
    TICKERS_DIR: Final[str] = './ml/tickers.txt'

    # Model Training Params
    LEARNING_RATE: Final[float] = 0.01
    BATCH_SIZE: Final[int] = 128
    EPOCHS: Final[int] = 500
    VALIDATION_SPLIT: Final[float] = 0.2
    PATIENCE: Final[int] = 20
    MIN_LR: Final[int] = 1e-8

    # Random Seed
    RANDOM: Final[int] = 42
