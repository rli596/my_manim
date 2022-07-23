###
# oneplusone.py
#
# Contains scenes for 1+n dimensions part of video
###

'''
Dependencies
'''

from venv import create
from manim import *
from numpy import number
# from sklearn.preprocessing import scale
from lorentz_utils import *

'''
Scenes
'''

class Metrics(Scene):
    def construct(self):
        dyn_text = MathTex(r'\text{Pseudo-distance} = -t^2 + x^2')
        one_plus_one_text = MathTex('-t^2 + x^2')

        one_plus_two_text = MathTex('-t^2 + x^2 + y^2')
        one_plus_three_text = MathTex('-t^2 + x^2 + y^2 + z^2')
        radial_text = MathTex('-t^2 + r^2')

        self.play(Write(dyn_text))
        self.wait()

        self.play(Transform(dyn_text, one_plus_three_text))
        self.wait()

        self.play(Transform(dyn_text, one_plus_two_text))
        self.wait()

        self.play(Transform(dyn_text, radial_text))
        self.wait()

        self.play(FadeOut(dyn_text))

class OneSheetHyperboloid(ThreeDScene):
    def construct(self):

        # Camera

        phi = 75
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        # Mobjects

        left_hyperbola = ParametricFunction(lambda xi: [-np.cosh(xi), 0, np.sinh(xi)], t_range = [-5, 5], color = RED)
        right_hyperbola = ParametricFunction(lambda xi: [np.cosh(xi), 0, np.sinh(xi)], t_range = [-5, 5], color = RED)
        hyperbolas = VGroup()
        hyperbolas.add(left_hyperbola, right_hyperbola)

        one_sheet_hyperboloid = Surface(
            lambda u,v: np.array([np.cosh(v) * np.cos(u), np.cosh(v) * np.sin(u), np.sinh(v)]),
            resolution = (16,16),
            u_range = [0, 2 * np.pi],
            v_range = [-3,3],
            fill_opacity = 0.7
        ).set_fill_by_checkerboard(RED,RED)

        # Animations

        self.play(FadeIn(hyperbolas))
        self.wait()
        self.play(Create(one_sheet_hyperboloid))
        self.wait()
        self.play(FadeOut(hyperbolas, one_sheet_hyperboloid))

class TwoSheetHyperboloid(ThreeDScene):
    def construct(self):

        # Camera

        phi = 75
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        # Mobjects

        upper_hyperbola = ParametricFunction(lambda xi: [np.sinh(xi), 0, np.cosh(xi)], t_range = [-5, 5], color = BLUE)
        lower_hyperbola = ParametricFunction(lambda xi: [np.sinh(xi), 0, -np.cosh(xi)], t_range = [-5, 5], color = BLUE)
        hyperbolas = VGroup()
        hyperbolas.add(upper_hyperbola, lower_hyperbola)

        upper_hyperboloid = Surface(
            lambda u,v: np.array([np.sinh(v) * np.cos(u), np.sinh(v) * np.sin(u), np.cosh(v)]),
            resolution = (16,16),
            u_range = [0, 2 * np.pi],
            v_range = [-3,3],
            fill_opacity = 0.5
        ).set_fill_by_checkerboard(BLUE,BLUE)

        lower_hyperboloid = Surface(
            lambda u,v: np.array([np.sinh(v) * np.cos(u), np.sinh(v) * np.sin(u), -np.cosh(v)]),
            resolution = (16,16),
            u_range = [0, 2 * np.pi],
            v_range = [-3,3],
            fill_opacity = 0.5
        ).set_fill_by_checkerboard(BLUE,BLUE)
        #two_sheet_hyperboloid = VGroup()
        #two_sheet_hyperboloid.add(upper_hyperboloid, lower_hyperboloid)

        # Animations

        self.play(FadeIn(hyperbolas))
        self.wait()
        self.play(Create(upper_hyperboloid), Create(lower_hyperboloid))
        self.wait()
        self.play(FadeOut(hyperbolas), FadeOut(upper_hyperboloid), FadeOut(lower_hyperboloid))

class LightCone(ThreeDScene):
    def construct(self):
        phi = 75
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        right_lightray = ParametricFunction(lambda xi: [xi, 0, xi], t_range = [-5, 5], color = PURPLE)
        left_lightray = ParametricFunction(lambda xi: [xi, 0, -xi], t_range = [-5, 5], color = PURPLE)
        lightrays = VGroup()
        lightrays.add(right_lightray, left_lightray)

        upper_light_cone = Surface(
            lambda u,v: np.array([v * np.cos(u), v * np.sin(u), v]),
            resolution = (16, 16),
            u_range = [0, 2 * np.pi],
            v_range = [0, 6],
            fill_opacity = 0.5
        ).set_fill_by_checkerboard(PURPLE, PURPLE)

        lower_light_cone = Surface(
            lambda u,v: np.array([v * np.cos(u), v * np.sin(u), -v]),
            resolution = (16, 16),
            u_range = [0,2 * np.pi],
            v_range = [0, 6],
            fill_opacity = 0.5
        ).set_fill_by_checkerboard(PURPLE, PURPLE)

        light_cones = VGroup()
        light_cones.add(upper_light_cone, lower_light_cone)

        # Animations

        self.play(FadeIn(lightrays))
        self.wait()
        self.play(Create(light_cones))
        self.wait()
        self.play(FadeOut(lower_light_cone, lightrays))
        self.wait()
        self.play(FadeOut(upper_light_cone))

class CelestialCircle(ThreeDScene):
    def construct(self):

        # Camera

        ## Set phi depending on scene required:
        ## phi = 75: skew
        ## phi = 0: top_down
        ## phi = 90: straight_on
        phi = 0
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        # Mobjects

        light_cone = Surface(
            lambda u,v: np.array([v * np.cos(u), v * np.sin(u), v]),
            resolution = (12, 12),
            u_range = [0,2 * np.pi],
            v_range = [0,12],
            fill_opacity = 0.5
        ).set_fill_by_checkerboard(PURPLE, PURPLE)

        time_slice = Surface(
            lambda u,v: np.array([u, v, 1]),
            resolution = (1,1),
            u_range = [-3, 3],
            v_range = [-3, 3],
            fill_opacity = 0.5,
        ).set_fill_by_checkerboard(GREEN, GREEN)

        celestial_circle = ParametricFunction(
            lambda t: np.array([np.cos(t), np.sin(t), 1]),
            t_range = [0, 2 * np.pi],
            color = YELLOW
        )

        stars = Mobject1D()
        n_stars = 12
        theta = 2 * np.pi / n_stars
        stars.add_points(
            [
                [np.cos(k * theta), np.sin(k * theta), 1]
                for k in range(n_stars)
            ]
        )

        # Transformations

        v = np.array([0.5, 0])
        Lambda = vec_to_boost_2(v)

        # Animations

        self.play(FadeIn(light_cone))
        # self.play(FadeIn(time_slice))
        self.wait()
        self.play(Create(celestial_circle))
        self.add(stars)
        # self.play(FadeOut(time_slice))
        self.wait()

        self.play(ApplyMatrix(Lambda, light_cone),
                  ApplyMatrix(Lambda, celestial_circle),
                  ApplyMatrix(Lambda, stars),
                  )
        self.wait()

        self.play(ApplyPointwiseFunction(lambda x: x/x[-1], celestial_circle),
                  ApplyPointwiseFunction(lambda x: x/x[-1], stars),
                  )
        self.wait()

class ConfCelestialCircle(ThreeDScene):
    def construct(self):

        # Camera

        ## Set phi depending on scene required:
        ## phi = 75: skew
        ## phi = 0: top_down
        ## phi = 90: straight_on
        phi = 0
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        # Mobjects

        celestial_circle = ParametricFunction(
            lambda t: np.array([np.cos(t), np.sin(t), 1]),
            t_range = [0, 2 * np.pi],
            color = YELLOW
        )

        stars = Mobject1D()
        n_stars = 12
        theta = 2 * np.pi / n_stars
        stars.add_points(
            [
                [np.cos(k * theta), np.sin(k * theta), 1]
                for k in range(n_stars)
            ]
        )

        # Transformations

        v_x = np.array([0.5, 0])
        Lambda_x = vec_to_boost_2(v_x)
        inv_Lambda_x = vec_to_boost_2(-v_x)

        v_y = np.array([0, 0.5])
        Lambda_y = vec_to_boost_2(v_y)
        inv_Lambda_y = vec_to_boost_2(-v_y)

        # Animations

        self.play(Create(celestial_circle))
        self.add(stars)
        self.wait()

        self.play(ApplyPointwiseFunction(lambda x: conformal_transformation_1_2(Lambda_x,x), stars))
        self.wait()
        self.play(ApplyPointwiseFunction(lambda x: conformal_transformation_1_2(inv_Lambda_x,x), stars))
        self.wait()
        self.play(ApplyPointwiseFunction(lambda x: conformal_transformation_1_2(Lambda_y,x), stars))
        self.wait()
        self.play(ApplyPointwiseFunction(lambda x: conformal_transformation_1_2(inv_Lambda_y,x), stars))
        self.wait()
        self.play(Rotate(stars, np.pi/6))
        self.wait()
        self.play(Rotate(stars, -np.pi/6))
        self.wait()
        self.remove(stars)
        self.play(FadeOut(celestial_circle))

class LorentzTransformation1_3(ThreeDScene):
    def construct(self):

        # Camera

        phi = 75
        theta = 15
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(1.8)

        # Mobjects
        celestial_sphere = Surface(
            lambda u, v: [np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)],
            resolution = (16, 16),
            u_range = [0, 2 * np.pi],
            v_range = [0, np.pi],
            fill_opacity = 0.5
        ).set_fill_by_checkerboard(PURPLE, PURPLE)

        stars = Mobject1D()
        star_positions = []
        n_U = 8
        n_V = 8
        for n in range(n_U):
            for m in range(n_V):
                U = n / n_U
                V = m / n_V
                u = 2 * np.pi * U
                v = np.pi * V
                star_positions.append([u, v])
        stars.add_points(
            [
                [np.cos(position[0]) * np.sin(position[1]), 
                np.sin(position[0]) * np.sin(position[1]),
                np.cos(position[1])]
                for position in star_positions
            ]
        )

        # Transformations

        v = [0, 0, -.8]
        Lambda = vec_to_boost_3(v)

        S = expm(SIGMA_Z * complex(0, 2/3 * np.pi))
        L_S = SL_to_SO(S)

        a = 1
        S_transl = np.array([[1, a],
                            [0, 1]])
        L_transl = SL_to_SO(S_transl)

        active_transformation = Lambda

        # Animations

        self.play(Create(celestial_sphere))
        self.wait()
        self.add(stars)
        self.wait()

        self.play(ApplyPointwiseFunction(lambda x: lorentz_transform_spacelike_proj(x, active_transformation), celestial_sphere),
                ApplyPointwiseFunction(lambda x: lorentz_transform_spacelike_proj(x, active_transformation), stars))
        self.play(ApplyPointwiseFunction(rescaling_3, celestial_sphere),
                ApplyPointwiseFunction(rescaling_3, stars))
        self.wait(1)

        self.remove(stars)
        self.play(FadeOut(celestial_sphere))

class RigidRotation(ThreeDScene):
    def construct(self):

        # Camera

        phi = 75
        theta = 15
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(1.8)

        # Mobjects
        celestial_sphere = Surface(
            lambda u, v: [np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)],
            resolution = (16, 16),
            u_range = [0, 2 * np.pi],
            v_range = [0, np.pi],
            fill_opacity = 0.5
        ).set_fill_by_checkerboard(PURPLE, PURPLE)

        stars = Mobject1D()
        star_positions = []
        n_U = 8
        n_V = 8
        for n in range(n_U):
            for m in range(n_V):
                U = n / n_U
                V = m / n_V
                u = 2 * np.pi * U
                v = np.pi * V
                star_positions.append([u, v])
        stars.add_points(
            [
                [np.cos(position[0]) * np.sin(position[1]), 
                np.sin(position[0]) * np.sin(position[1]),
                np.cos(position[1])]
                for position in star_positions
            ]
        )

        # Transformations

        '''
        v = [0, 0, -1]
        Lambda = vec_to_boost_3(v)
        '''

        # Animations

        self.play(Create(celestial_sphere))
        self.wait()
        self.add(stars)
        self.wait()

        self.play(Rotate(celestial_sphere, angle = np.pi/8, about_point = ORIGIN, rate_func =linear),
                Rotate(stars, angle = np.pi/8, about_point = ORIGIN, rate_func =linear))
        self.wait(1)

        self.remove(stars)
        self.play(FadeOut(celestial_sphere))

class ConformalTransformation1_3(ThreeDScene):
    def construct(self):

        # Camera

        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(1.8)

        # Mobjects
        celestial_sphere = Surface(
            lambda u, v: [np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)],
            resolution = (16, 16),
            u_range = [0, 2 * np.pi],
            v_range = [0, np.pi],
            fill_opacity = 0.5
        ).set_fill_by_checkerboard(PURPLE, PURPLE)

        stars = Mobject1D()
        star_positions = []
        n_U = 8
        n_V = 8
        for n in range(n_U):
            for m in range(n_V):
                U = n / n_U
                V = m / n_V
                u = 2 * np.pi * U
                v = np.pi * V
                star_positions.append([u, v])
        stars.add_points(
            [
                [np.cos(position[0]) * np.sin(position[1]), 
                np.sin(position[0]) * np.sin(position[1]),
                np.cos(position[1])]
                for position in star_positions
            ]
        )

        # Transformations

        v_z = np.array([0, 0, -0.8])
        Lambda_z = vec_to_boost_3(v_z)
        inv_Lambda_z = vec_to_boost_3(-v_z)

        v_x = np.array([-0.8, 0, 0])
        Lambda_x = vec_to_boost_3(v_x)
        inv_Lambda_x = vec_to_boost_3(-v_x)

        S = np.array([[1,1], [0,1]])
        Lambda = SL_to_SO(S)
        inv_Lambda = SL_to_SO(np.array([[1,-1],[0,1]]))

        # Animations

        self.play(Create(celestial_sphere))
        self.add(stars)
        self.wait(1)
        self.play(ApplyPointwiseFunction(lambda x: conformal_transformation_3_1(x, Lambda_x), celestial_sphere),
                ApplyPointwiseFunction(lambda x: conformal_transformation_3_1(x, Lambda_x), stars))
        self.wait(1)
        self.play(ApplyPointwiseFunction(lambda x: conformal_transformation_3_1(x, inv_Lambda_x), celestial_sphere),
                ApplyPointwiseFunction(lambda x: conformal_transformation_3_1(x, inv_Lambda_x), stars)) 
        self.wait(1)

        self.play(Rotate(celestial_sphere, angle = np.pi/8, about_point = ORIGIN, rate_func =linear),
                Rotate(stars, angle = np.pi/8, about_point = ORIGIN, rate_func =linear))
        self.wait(1)

        self.play(Rotate(celestial_sphere, angle = -np.pi/8, about_point = ORIGIN, rate_func =linear),
                Rotate(stars, angle = np.pi/8, about_point = ORIGIN, rate_func =linear))
        self.wait(1)

        self.play(ApplyPointwiseFunction(lambda x: conformal_transformation_3_1(x, Lambda), celestial_sphere),
                ApplyPointwiseFunction(lambda x: conformal_transformation_3_1(x, Lambda), stars))
        self.wait(1)
        self.play(ApplyPointwiseFunction(lambda x: conformal_transformation_3_1(x, inv_Lambda), celestial_sphere),
                ApplyPointwiseFunction(lambda x: conformal_transformation_3_1(x, inv_Lambda), stars))
        self.wait(1)

        
        self.remove(stars)
        self.play(FadeOut(celestial_sphere))

class Text(Scene):
    def construct(self):
        dyn_text = MathTex(r'\text{Spacetime symmetries}\rightarrow \text{Complex geometry?}')

        self.play(Write(dyn_text))
        self.wait()

        self.play(FadeOut(dyn_text))