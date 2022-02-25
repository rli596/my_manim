####
# cx_transform.py
#
# Script to animate complex transformations
####

'''
Dependencies
'''
from manim import *
import numpy as np

'''
Functions
'''

def cx_fn(z):
    f = z + 1/z
    return f

def mobius_map(z):
    a = 0
    b = 1
    c = 1
    d = 0
    f = (a * z + b)/(c * z + d)
    return f

'''
Scene
'''

class ComplexTransform(Scene):
    def construct(self):
        # Load Mobjects
        grid1 = NumberPlane()
        grid2 = NumberPlane()
        circle = Circle(radius=2)
        circle.set_fill(PINK, opacity=0.5)

        # Create Mobjects in scene
        self.add(grid1, grid2)
        self.play(
            Create(grid1, run_time=3, lag_ratio=0.1),
            Create(grid2, run_time=3, lag_ratio=0.1),
            Create(circle, run_time=3, lag_ratio=0.1)
        )
        self.wait()

        # Map animation
        self.play(
            circle.animate.apply_complex_function(
                mobius_map
            ),
            run_time=3
        )
        self.wait()