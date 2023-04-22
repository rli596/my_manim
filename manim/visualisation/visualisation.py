###
#
# visualisation.py
#
# python script containing scenes for visualisation video
#
###

'''
Dependencies
'''

from manim import *

'''
Scenes
'''

class Lines(ThreeDScene):
    def construct(self):
        # Camera
        phi = 0
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        # Mobjects
        axes = NumberPlane()
        f_id = ParametricFunction(lambda t: np.array([t,t,0]), 
                                  t_range = [-8, 8],
                                  color = RED)
        f_id_text = MathTex('f(x) = x').set_x(-5).set_y(+2.5).set_color(YELLOW)
        f_1_text = MathTex(r'f(x) = \frac{x}{3} - 1').set_x(-5).set_y(+2.5).set_color(YELLOW)
        # Animations
        self.play(FadeIn(axes))
        self.wait()
        self.play(Write(f_id_text))
        self.wait()
        self.play(Create(f_id))
        self.wait()
        self.play(Transform(f_id_text, f_1_text))
        self.wait()
        self.play(Homotopy(lambda x,y,z,t: np.array([x, y * (1-t) + (x / 3 - 1) * t, z]), f_id))
        self.wait()