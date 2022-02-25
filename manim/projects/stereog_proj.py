
'''
Dependencies
'''
from manim import *

'''
Constants
'''
delta_theta = np.pi/30 # To avoid divide by zero
resolution_fa = 16

'''
Functions
'''

#Parametric surfaces
def param_sphere(u, v):
            R = 1
            x = R * np.cos(u) * np.sin(v)
            y = R * np.sin(u) * np.sin(v)
            z = R * np.cos(v)
            return np.array([x, y, z])

def param_plane(u, v):
            R = 1
            x_0 = R * np.cos(u) * np.sin(v)
            y_0 = R * np.sin(u) * np.sin(v)
            z_0 = R * np.cos(v)
            x = x_0/(1-z_0)
            y = y_0/(1-z_0)
            z = 0
            return np.array([x, y, z])

'''
Scenes
'''

class Stereographic(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=-90 * DEGREES)

        # Load in surfaces
            # Static
        sphere = Surface(
            param_sphere,
            resolution=(resolution_fa, resolution_fa),
            u_range=[-np.pi, np.pi],
            v_range=[0 + delta_theta, np.pi],
        )

        sphere_south_pole = Surface(
            param_sphere,
            resolution=(resolution_fa, resolution_fa),
            u_range=[-np.pi, np.pi],
            v_range=[np.pi - delta_theta, 0],
        )

        plane = Surface(
            param_plane,
            resolution=(resolution_fa, resolution_fa),
            u_range=[-np.pi, np.pi],
            v_range=[0 + delta_theta, np.pi],
        )

            # Initialise transforming surface
        anim_surface = Surface(
            param_sphere,
            resolution=(resolution_fa, resolution_fa),
            u_range=[-np.pi, np.pi],
            v_range=[0 + delta_theta, np.pi],
        )

        # Set surface colouring and opacity
        surfaces = [sphere,
        sphere_south_pole,
        plane,
        anim_surface]

        for surface in surfaces:
            surface.set_fill_by_checkerboard(ORANGE, BLUE, opacity = 0.5)

        # Animation
        axes = ThreeDAxes()
        self.add(axes, anim_surface)
        self.play(
            Transform(anim_surface, plane)
        )
        self.wait()
        self.play(
            Transform(anim_surface, sphere_south_pole)
        )
        self.wait()
        self.play(
            Transform(anim_surface, plane)
        )
        self.wait()
        self.play(
            Transform(anim_surface, sphere)
        )
        self.wait()

class StereographicAtlas(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=-90 * DEGREES)

        # Load surfaces
            # Static
        sphere = Surface(
        param_sphere,
        resolution=(resolution_fa, resolution_fa),
        u_range=[-np.pi, np.pi],
        v_range=[0 + delta_theta, np.pi],
        )
        sphere.set_fill(BLUE, opacity=0.5)

        theta_0 = np.pi/3

        sphere_patch_1 = Surface(
        param_sphere,
        resolution=(resolution_fa, resolution_fa),
        u_range=[-np.pi, np.pi],
        v_range=[0 + theta_0, np.pi],
        )
        sphere_patch_1.set_fill(RED, opacity=0.5)

        sphere_patch_2 = Surface(
        param_sphere,
        resolution=(resolution_fa, resolution_fa),
        u_range=[-np.pi, np.pi],
        v_range=[np.pi - theta_0, 0],
        )
        sphere_patch_2.set_fill(GREEN, opacity=0.5)

            # Animated
        anim_patch_1 = Surface(
        param_plane,
        resolution=(resolution_fa, resolution_fa),
        u_range=[-np.pi, np.pi],
        v_range=[0 + theta_0, np.pi],
        )
        anim_patch_1.set_fill(RED, opacity=0.5)

        anim_patch_2 = Surface(
        param_plane,
        resolution=(resolution_fa, resolution_fa),
        u_range=[-np.pi, np.pi],
        v_range=[0 + theta_0, np.pi],
        )
        anim_patch_2.set_fill(GREEN, opacity=0.5)

            # Create patch and map to sphere
        self.play(FadeIn(anim_patch_1))
        self.play(Transform(anim_patch_1, sphere_patch_1))

            # Create patch and map to sphere
        self.play(FadeIn(anim_patch_2))
        self.play(Transform(anim_patch_2, sphere_patch_2))

            # Fade out
        self.play(FadeOut(anim_patch_2))
        self.play(FadeOut(anim_patch_1))