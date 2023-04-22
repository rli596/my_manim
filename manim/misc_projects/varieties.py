###
#
# varieties.py
#
###

'''
Dependencies
'''

import numpy as np
from manim import *

'''
Scenes
'''

class CubicSurface(ThreeDScene):
    def construct(self):

        # Camera

        phi = 75
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(.5)

        # Mobjects
        cubic_surface = Surface(lambda u,v: np.array([u, v, np.cbrt(1 - u**3 + v**3)]),
                                u_range = [-5,5],
                                v_range = [-5,5],
                                resolution = (24,24),
                                fill_opacity=0.5)

        # Animations
        self.add(cubic_surface)
        self.begin_ambient_camera_rotation(rate = np.pi/3, about = "theta")
        self.wait(6)

class InvariantCubic(ThreeDScene):
    def construct(self):

        # Camera

        phi = 30
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(.5)

        # Mobjects
        range = 3
        cubic_surface = Surface(lambda u,v: np.array([u, v, v**3/4 - 3 * u**2 * v /4]),
                                u_range = [-range,range],
                                v_range = [-range,range],
                                resolution = (16,16),
                                fill_opacity=0.5)

        # Animations
        self.add(cubic_surface)