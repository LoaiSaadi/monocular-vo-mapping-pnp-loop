from dataclasses import dataclass
import numpy as np
from pose import Pose


@dataclass
class FRAME:
    id: int
    timestamp: float
    pose: Pose
    key_points: list
    descriptors: np.ndarray
    image_gray: np.ndarray
    image_bgr: np.ndarray
    processed: bool = True