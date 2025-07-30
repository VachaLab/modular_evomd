# === residue.py ===
import numpy as np
import logging
from scales import Scales

logger = logging.getLogger(__name__)

class Residue:
    name = 'residue'
    increment = -1  # used to compute height
    theta = 100  # used to compute position on a unit circle

    def __init__(self, letter, index, scale) -> None:
        self.letter = letter
        self.index = index
        self.charge = Scales.aa_charges[self.letter]
        self.hydrophobicity = Scales.hydrophobicity_scales[scale][self.letter]
        self.position = self._compute_position()
        self.set_rotated_position = self.position
    
    # special methods -------------------------------
    def __str__(self):
        return self.letter
    
    # set properties ----
    def _compute_position(self):
        """
        Computes 3D position and returns
        """
        rad = np.deg2rad(self.theta)
        angle = self.index * rad
        height = self.index * self.increment
        cos = np.cos(angle)
        sin = np.sin(angle)
        return np.array([cos, sin, height])
    
    def set_new_position(self, new_position):
        self.position = new_position
    
    # get information ----
    def get_hm_contribution(self):
        xy_plane_moment = self.hydrophobicity * self.position[:2]
        xyz_cont = np.array([xy_plane_moment[0], xy_plane_moment[1], 0.])
        return xyz_cont
    
    @property
    def x(self):
        return self.position[0]
    
    @property
    def y(self):
        return self.position[1]
    
    @property
    def z(self):
        return self.position[2]

if __name__ == '__main__':
    pass
