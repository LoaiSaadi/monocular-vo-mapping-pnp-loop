from dataclasses import dataclass
import numpy as np


@dataclass
class Pose:
    t_world_cam: np.ndarray
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray

    @staticmethod
    def from_t(t_world_cam: np.ndarray) -> "Pose":
        t_world_cam = t_world_cam.astype(np.float64)
        r = t_world_cam[:3, :3].copy()
        t = t_world_cam[:3, 3:4].copy()
        return Pose(t_world_cam=t_world_cam, rotation_matrix=r, translation_vector=t)

    def set_t(self, t_world_cam: np.ndarray) -> None:
        self.t_world_cam = t_world_cam.astype(np.float64)
        self.rotation_matrix = self.t_world_cam[:3, :3].copy()
        self.translation_vector = self.t_world_cam[:3, 3:4].copy()