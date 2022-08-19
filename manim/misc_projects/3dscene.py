###
# 3dscene.py
#
# Playing around with 3d animation capabilities. 
# Learnt that set_camera_orientation variables are opposite to standard labelling
# of polar and azimuthal angles by theta and phi.

'''
Dependencies
'''
from math import radians
from manim import *
import numpy as np
from functools import partial

'''
Functions
'''
# Parametrised surfaces
def param_gauss(u, v):
            x = u
            y = v
            sigma, mu = 0.4, [0.0, 0.0]
            d = np.linalg.norm(np.array([x - mu[0], y - mu[1]]))
            z = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
            return np.array([x, y, z])

def param_torus(u, v):
    r = 1
    R = 2
    x = R * np.cos(u) + r * np.cos(u) * np.cos(v)
    y = R * np.sin(u) + r * np.sin(u) * np.cos(v)
    z = r * np.sin(v)
    return np.array([x, y, z])

def param_cylinder(u, v):
    r = 1
    L = 1
    x = r * np.cos(v)
    y = L * u
    z = r * np.sin(v)
    return np.array([x, y, z])

def param_cylinder_twisted(u, v):
    r = 1
    L = 1
    x = r * np.cos(v + u)
    y = L * u
    z = r * np.sin(v + u)
    return np.array([x, y, z])

def param_cylinder_twist_angle(t, u, v):
    r = 1
    L = 1
    x = r * np.cos(v + t*u)
    y = L * u
    z = r * np.sin(v + t*u)
    return np.array([x, y, z])

def param_torus_twisted(u, v):
    r = 1
    R = 2
    x = R * np.cos(u) + r * np.cos(u) * np.cos(v + u)
    y = R * np.sin(u) + r * np.sin(u) * np.cos(v + u)
    z = r * np.sin(v + u)
    return np.array([x, y, z])

def param_klein_bottle(u, v):
    cu = np.cos(u)
    su = np.sin(u)
    cv = np.cos(v)
    sv = np.sin(v)

    x = (-2/15) * cu * (3*cv - 30*su + 90*cu**4*su - 60*cu**6*su + 5 * cu * cv * su)
    y = (-1/15) * su * (3*cv - 3*cu**2*cv - 48*cu**4*cv + 48*cu**6*cv \
        + 60*su + 5*cu*cv*su - 5*cu**3*cv*su - 80*cu**5*cv*su + 80*cu**7*cv*su)
    z = (2/15)*(3 + 5*cu*su)*sv
    return np.array([x, y, z])

class ThreeDGaussianPlot(ThreeDScene):
    def construct(self):

        # Camera

        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)

        # Mobjects

        resolution_fa = 42
        gauss_plane = Surface(
            param_gauss,
            resolution=(resolution_fa, resolution_fa),
            v_range=[-2, +2],
            u_range=[-2, +2]
        )

        gauss_plane.scale(2, about_point=ORIGIN)
        gauss_plane.set_style(fill_opacity=1,stroke_color=GREEN)
        gauss_plane.set_fill_by_checkerboard(ORANGE, BLUE, opacity=0.5)
        axes = ThreeDAxes()

        # Animations

        self.add(axes,gauss_plane)
        self.begin_3dillusion_camera_rotation(rate=2)
        self.wait(PI/2)
        self.stop_3dillusion_camera_rotation()

class ThreeDTorusPlot(ThreeDScene):
    def construct(self):

        # Camera

        self.set_camera_orientation(phi=75 * DEGREES, theta=-60 * DEGREES)

        # Mobjects

        resolution_fa = 42
        torus = Surface(
            param_torus,
            resolution=(resolution_fa, resolution_fa),
            u_range=[-np.pi, np.pi],
            v_range=[-np.pi, np.pi],
        )

        # torus.scale(2, about_point=ORIGIN)
        torus.set_fill_by_checkerboard(ORANGE, BLUE, opacity=0.5)
        axes = ThreeDAxes()

        # Animations

        self.play(Create(torus))

class ThreeDTorusAnim(ThreeDScene):
    def construct(self):

        # Camera
        
        self.set_camera_orientation(phi=75 * DEGREES, theta=-60 * DEGREES)

        # Mobjects
        resolution_fa = 42
        torus = Surface(
            param_torus,
            resolution=(resolution_fa, resolution_fa),
            u_range=[-np.pi, np.pi],
            v_range=[-np.pi, np.pi],
        )

        cylinder = Surface(
            param_cylinder,
            resolution=(resolution_fa, resolution_fa),
            u_range=[-np.pi, np.pi],
            v_range=[-np.pi, np.pi],
        )

        # torus.scale(2, about_point=ORIGIN)
        torus.set_fill_by_checkerboard(ORANGE, BLUE, opacity=0.5)
        cylinder.set_fill_by_checkerboard(ORANGE, BLUE, opacity=0.5)
        axes = ThreeDAxes()

        # Animations

        self.add(axes, torus)
        self.play(
            Transform(torus, cylinder)
        )

class DehnTwist(ThreeDScene):
    def construct(self):

        # Camera 
        self.set_camera_orientation(phi=75 * DEGREES, theta=-60 * DEGREES)

        # Mobjects
        
        resolution_fa = 32 # 42 for nice resolution

        torus = Surface(
            param_torus,
            resolution=(resolution_fa, resolution_fa),
            u_range=[-np.pi, np.pi],
            v_range=[-np.pi, np.pi],
        )

        cylinder = Surface(
            param_cylinder,
            resolution=(resolution_fa, resolution_fa),
            u_range=[-np.pi, np.pi],
            v_range=[-np.pi, np.pi],
        )

        twisted_cylinder = Surface(
            param_cylinder_twisted,
            resolution=(resolution_fa, resolution_fa),
            u_range=[-np.pi, np.pi],
            v_range=[-np.pi, np.pi],
        )

        twisted_torus = Surface(
            param_torus_twisted,
            resolution=(resolution_fa, resolution_fa),
            u_range=[-np.pi, np.pi],
            v_range=[-np.pi, np.pi],
        )

        twist_surface = Surface(
            param_torus,
            resolution=(resolution_fa, resolution_fa),
            u_range=[-np.pi, np.pi],
            v_range=[-np.pi, np.pi],
        )

        surfaces = [torus, cylinder, twisted_cylinder, twisted_torus, twist_surface]

        # torus.scale(2, about_point=ORIGIN)
        for surface in surfaces:
            surface.set_fill_by_checkerboard(ORANGE, BLUE, opacity=1)

        # Animations

        axes = ThreeDAxes()
        self.add(axes, twist_surface)
        self.play(Transform(twist_surface, cylinder, run_time=3))
        self.play(Transform(twist_surface, twisted_cylinder, run_time=3))    
        self.play(Transform(twist_surface, twisted_torus, run_time=3))
        self.play(Transform(twist_surface, torus, run_time=3))

class KleinBottle(ThreeDScene):
    def construct(self):

        # Camera

        self.set_camera_orientation(phi=phi_degs * DEGREES, theta=theta_degs * DEGREES)

        # Mobjects

        resolution_fa = 16
        phi_degs = 45
        theta_degs = 45

        kb = Surface(
            param_klein_bottle,
            resolution=(resolution_fa, resolution_fa),
            u_range=[0, np.pi],
            v_range=[0, 2*np.pi],
        )

        # Animations

        # torus.scale(2, about_point=ORIGIN)
        kb.set_fill_by_checkerboard(ORANGE, BLUE, opacity=0.5)
        axes = ThreeDAxes()
        self.add(axes)
        self.play(Create(kb))
        self.wait()
        #self.play(FadeOut(kb))
        #self.wait()

class RotateCylinder(ThreeDScene):
    def construct(self):

        # Camera

        phi_degs = 75
        theta_degs = 30
        self.set_camera_orientation(phi = phi_degs * DEGREES, 
        theta = theta_degs * DEGREES)

        # Mobjects

        resolution_fa = 12
        
        twisted_cyls = {}

        ts = np.arange(0, 1, 1/6)

        for t in ts:
            twisted_cyls[t] = Surface(
                partial(param_cylinder_twist_angle, t),
                resolution=(resolution_fa, resolution_fa),
                u_range = [-np.pi, np.pi],
                v_range = [-np.pi, np.pi]
            )

        anim_surface = Surface(
            param_cylinder,
            resolution=(resolution_fa, resolution_fa),
                u_range = [-np.pi, np.pi],
                v_range = [-np.pi, np.pi]
            )

        self.add(anim_surface)

        # Animations
        
        for t in ts:
            self.play(Transform(anim_surface, twisted_cyls[t]),
            )
