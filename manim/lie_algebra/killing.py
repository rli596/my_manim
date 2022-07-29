###
#
# killing.py
#
# Killing vector fields
#
###

'''
Dependencies
'''

from manim import *
from utils import *
from functools import partial

'''
Scenes
'''

class SphereZRot(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi = 60 * DEGREES, theta = 15*DEGREES)
        self.camera.set_zoom(2)

        # Mobjects

        sphere = Surface(
            lambda phi, theta: np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]),
            resolution=(12, 12),
            u_range = [0, 2 * np.pi],
            v_range = [0, np.pi],
            fill_opacity= 0.7
        ).set_fill_by_checkerboard(BLUE, BLUE)

        vector_field = VGroup()
        phis = np.arange(0, 2 * np.pi, np.pi/6)
        thetas = np.arange(0, np.pi, np.pi/12)
        for phi in phis:
            for theta in thetas:
                x = np.cos(phi) * np.sin(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(theta)

                basepoint = np.array([x,y,z])
                vec_array = np.array([-y, x, 0])/3
                vec_base = basepoint + 0.5 * vec_array
                vector = Vector(vec_array,
                                color = interpolate_color(PURPLE, YELLOW, (basepoint[2] + 1)/2)
                                ).set_x(vec_base[0]).set_y(vec_base[1]).set_z(vec_base[2])
                vector_field.add(vector)


        self.add(sphere, vector_field)
        self.play(Rotate(sphere, np.pi/6, axis = np.array([0,0,1])),
                  Rotate(vector_field, np.pi/6, axis = np.array([0,0,1])))

class SCT(ThreeDScene):
    def construct(self):

        # Camera
        self.camera.set_zoom(2)

        # Mobjects

        x_bound = 4
        y_bound = 3.4
        spacing = 0.2
        shift = 1
        radius = 3.1

        half_plane = Surface(
            lambda x, y: np.array([x, y-shift, 0]),
            resolution = (int(2 * x_bound / spacing), int(y_bound / spacing)),
            u_range = [-x_bound, +x_bound],
            v_range = [0, +y_bound],
            fill_opacity= 0.7,
        ).set_fill_by_checkerboard(BLUE, BLUE)

        vector_field = VGroup()
        xs = np.arange(-x_bound, +x_bound, spacing)
        ys = np.arange(0, +y_bound, spacing)
        for x in xs:
            for y in ys:
                if x**2 + y**2 < radius**2:
                    basepoint = np.array([x,y,0])
                    vec_array = np.array([x**2 - y**2, 2*x*y,0])/13
                    vec_base = basepoint - np.array([0,shift,0])
                    vector = Vector(vec_array,
                    color = interpolate_color(YELLOW, PURPLE, (x**2 + y**2)**2/radius**4)).set_x(vec_base[0]).set_y(vec_base[1])
                    vector_field.add(vector)

        self.add(half_plane, vector_field)