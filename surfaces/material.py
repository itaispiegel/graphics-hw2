from typing import List

import numpy as np


class Material:
    def __init__(
        self,
        diffuse_color: List[float],
        specular_color: List[float],
        reflection_color: List[float],
        shininess: float,
        transparency: float,
    ):
        self.diffuse_color = np.array(diffuse_color)
        self.specular_color = np.array(specular_color)
        self.reflection_color = reflection_color
        self.shininess = shininess
        self.transparency = transparency
