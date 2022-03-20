###
# conformal.py
#
# Contains manim scenes for conformal group animations
###

'''
Dependencies
'''

from manim import *
from conformal_utils import *

'''
Scenes
'''

class ConformalTransformation(ThreeDScene):
    def construct(self):
        phi = 45
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        # Mobjects

        celestial_torus = Surface(
            embedded_torus,
            resolution = (16,16),
            u_range = (-np.pi, np.pi),
            v_range = (-np.pi, np.pi),
            fill_opacity = 0.5
        ).set_fill_by_checkerboard(PURPLE, PURPLE)

        stars = Mobject1D()
        star_angles = []
        root_num_stars = 9
        for u in np.linspace(-np.pi, np.pi, root_num_stars):
            for v in np.linspace(-np.pi, np.pi, root_num_stars):
                star_angles.append(np.array([u, v]))
        stars.add_points(
            [
                embedded_torus(position[0], position[1])
                for position in star_angles
            ]
        )

        # Transformations

        Lambda = expm(K_11)

        # Animations

        self.add(celestial_torus, stars)
        self.play(ApplyPointwiseFunction(lambda y: conformal_transformation(y, Lambda), stars),
        ApplyPointwiseFunction(lambda y: conformal_transformation(y, Lambda), celestial_torus))

class ConformalTransformationFundPoly(ThreeDScene):
    def construct(self):
        phi = 0
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        # Mobjects

        fund_poly = Surface(
            lambda u,v: np.array([u, v, 0]),
            resolution = (16,16),
            u_range = (-np.pi, np.pi),
            v_range = (-np.pi, np.pi),
            fill_opacity = 0.5
        ).set_fill_by_checkerboard(PURPLE, PURPLE)

        stars = Mobject1D()
        star_angles = []
        root_num_stars = 17
        for u in np.linspace(-np.pi, np.pi, root_num_stars):
            for v in np.linspace(-np.pi, np.pi, root_num_stars):
                star_angles.append(np.array([u, v]))
        stars.add_points(
            [
                np.array([position[0], position[1], 0])
                for position in star_angles
            ]
        )

        # Transformations

        Lambda = expm(K_11)

        # Animations

        self.add(fund_poly, stars)
        self.play(ApplyPointwiseFunction(lambda y: np.pad(conformal_tfmn_fundamental_polygon(y[0:2], Lambda), (0,1)), stars),
        ApplyPointwiseFunction(lambda y: np.pad(conformal_tfmn_fundamental_polygon(y[0:2], Lambda), (0,1)), fund_poly)
        )