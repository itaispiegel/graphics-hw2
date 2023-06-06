from typing import List

import numpy as np


class Light:
    def __init__(
        self,
        position: List[float],
        color: List[float],
        specular_intensity: float,
        shadow_intensity: float,
        radius: float,
    ):
        self.position = np.array(position)
        self.color = np.array(color)
        self.specular_intensity = specular_intensity
        self.shadow_intensity = shadow_intensity
        self.radius = radius
