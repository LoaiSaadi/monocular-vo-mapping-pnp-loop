from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Map:
    points: list = field(default_factory=list)
    frames: list = field(default_factory=list)

    _dirty: bool = True
    _desc_cache: Optional[np.ndarray] = None
    _xyz_cache: Optional[np.ndarray] = None

    def mark_dirty(self) -> None:
        self._dirty = True

    def rebuild_cache(self) -> None:
        if not self._dirty:
            return
        if len(self.points) == 0:
            self._desc_cache = np.zeros((0, 32), dtype=np.uint8)
            self._xyz_cache = np.zeros((0, 3), dtype=np.float64)
        else:
            self._desc_cache = np.vstack([p.descriptor for p in self.points]).astype(np.uint8)
            self._xyz_cache = np.vstack([p.point for p in self.points]).astype(np.float64)
        self._dirty = False