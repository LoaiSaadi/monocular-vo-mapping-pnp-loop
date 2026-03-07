from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Point:
    id: int
    point: np.ndarray
    frames: list = field(default_factory=list)
    descriptor: Optional[np.ndarray] = None