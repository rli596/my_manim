####
# joukowsky.py
#
# Script to animate a circle transforming under the joukowsky transform and back
####

'''
Dependencies
'''
from manim import *
import numpy as np

'''
Functions
'''

def joukowsky_map(z):
    f = z + 1/z
    return f

'''
Configuration
'''

'''
Scene
'''

class Joukowsky(Scene):
    def construct(self):
        # Load Mobjects
        grid = NumberPlane()
        circle = Circle(radius=1.08).shift(LEFT*0.08).shift(UP*0.08)
        circle_fixed = Circle(radius=1.08).shift(LEFT*0.08).shift(UP*0.08)

        # Create Mobjects in scene
        self.add(grid)
        self.add(circle)

        # Map animation
        self.play(
            circle.animate.apply_complex_function(
                joukowsky_map
            ),
            
        )
        self.wait()

        self.play(
            Transform(circle, circle_fixed)
        )
        self.wait()