###
#
# wikipedia.py
#
# python script containing scenes for Wikipedia
#
###

'''
Dependencies
'''

from manim import *
import scipy as sp

'''
Functions
'''

def sombrero_function(r):
    f = (r**2 - 1**2)**2
    return f

'''
Scenes
'''

class MexicanHat(ThreeDScene):
    def construct(self):
        # Camera
        phi = 60
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(3)

        # Mobjects
        f = Surface(lambda u,v: np.array([u * np.cos(v), u * np.sin(v), sombrero_function(u)]),
                    u_range = [0, 1.3],
                    v_range = [0, 2 * np.pi],
                    resolution=(24,12),
                    fill_opacity=0.7)

        c = ParametricFunction(lambda t: np.array([np.cos(t), np.sin(t), 0]),
                                t_range = [0, 2 * np.pi],
                                color = YELLOW,
                                )
        # Animations
        self.add(c, f)

class WittVectorFields(Scene):
    def construct(self):
        n = -2
        interval = 0.5
        x_range = 7
        y_range = 4
        def func(pos):
            z = complex(pos[0],pos[1])
            f = z**n
            return np.array([np.real(f), np.imag(f), 0])
        vector_field = ArrowVectorField(func, x_range = [-x_range - interval/2, x_range + interval/2, interval], y_range = [- y_range - interval/2, y_range + interval/2, interval])
        self.add(vector_field)

class WittVectorFields3D(ThreeDScene):
    
    def construct(self):

        # Camera
        phi = 0
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(2)


        n = -3
        interval = 0.4
        x_range = 6
        y_range = 6
        def func(pos):
            z = complex(pos[0],pos[1])
            f = -z**(n+1)
            return np.array([np.real(f), np.imag(f), 0])/2
        vector_field = ArrowVectorField(func, x_range = [-x_range - interval/2, x_range + interval/2, interval], y_range = [-y_range - interval/2, y_range + interval/2, interval])
        self.add(vector_field)

class SeibergWittenKahlerMetric(ThreeDScene):
    def construct(self):
        # Camera
        phi = 0
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(0.5)

        # Mobjects
        omega_squared = lambda u: 2/(1 + u)
        nu_squared = lambda u: (u - 1)/(1 + u)
        K = lambda omega_squared: np.pi / 2 * sp.special.hyp2f1(1/2, 1/2, 1, omega_squared)
        tau = lambda omega_squared, nu_squared: i * K(nu_squared)/ K(omega_squared)
        metric = lambda u: np.imag(tau(omega_squared(u), nu_squared(u)))
        graph_mobject = Surface(lambda x,y: np.array([x , y, metric(complex(x, y))]),
                    u_range = [-4, 4],
                    v_range = [1, 4],
                    resolution=(12,12),
                    fill_opacity=0.7)

        # Animations
        self.add(graph_mobject)