import numpy as np


class Surface:
    def __init__(self, material_index: int):
        self.material_index = material_index

    def intersect(self, source: np.ndarray, ray_vec: np.ndarray):
        """
        Return the intersection point and distance from the source to the intersection point.
        """
        raise NotImplementedError()
