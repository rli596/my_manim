###
# lie_algebra.py
#
# Visualising left invariant vector fields in the simple U(1) case
###

'''
Dependencies
'''

from manim import *

'''
Constants
'''
N_THETAS = 12
ARROW_LENGTH = 0.7

'''
Functions
'''
def param_circle(t):
    '''
    Parameters
    ----------
    t: double
        parameter
    Returns
    -------
    p: point parametrised by t
    '''
    R = 1
    x = R*np.cos(t)
    y = R*np.sin(t)
    return np.array([x,y,0])

'''
Scenes
'''
class StaticVectorField(Scene):
    def construct(self):
        circle = ParametricFunction(param_circle, [-np.pi, np.pi])

        v_field = VGroup()
        for theta in np.linspace(0, 2*np.pi, 7):
            n = np.array([np.cos(theta), np.sin(theta)])
            t = np.array([-np.sin(theta), np.cos(theta)])

            v_field += Arrow(buff = ARROW_LENGTH).shift(
                RIGHT*(n[0]+ ARROW_LENGTH*t[0]/2) 
                + UP*(n[1]+ARROW_LENGTH*t[1]/2)).rotate(
                    theta + np.pi/2)
        
        self.add(circle, v_field)

class AnimVectorField(Scene):
    def construct(self):
        circle = ParametricFunction(param_circle, [-np.pi, np.pi]).set_color(
            BLUE
        )

        v_field = VGroup()
        for theta in np.linspace(0, 2*np.pi*(1 - N_THETAS), N_THETAS):
            n = np.array([np.cos(theta), np.sin(theta)])
            t = np.array([-np.sin(theta), np.cos(theta)])

            v_field += Arrow(buff = ARROW_LENGTH).shift(
                RIGHT*(n[0]+ ARROW_LENGTH*t[0]/2) 
                + UP*(n[1]+ARROW_LENGTH*t[1]/2)).rotate(
                    theta + np.pi/2)
        
        self.add(circle, v_field)
        self.play(Rotate(v_field, angle = np.pi/3))