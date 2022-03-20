###
# lorentz.py
#
# Contains manim scenes for lorentz group animations
###

'''
Dependencies
'''

from manim import *
from lorentz_utils import *

'''
Scenes
'''



class LorentzTransformation1_2(ThreeDScene):
    def construct(self):
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        # Mobjects

        light_cone_bottom = Surface(
            light_cone,
            resolution = (16, 16),
            u_range = [0,1],
            v_range = [0, 2 * np.pi],
            fill_opacity = 0.5,
            fill_color = GREEN
        ).set_fill_by_checkerboard(RED, RED)

        light_cone_top = Surface(
            light_cone,
            resolution = (16, 16),
            u_range = [1,4],
            v_range = [0, 2 * np.pi],
            fill_opacity = 0.5,
            fill_color = GREEN
        ).set_fill_by_checkerboard(BLUE, BLUE)

        time_slice = Surface(
            lambda u,v: np.array([u, v, 1]),
            resolution = (1,1),
            u_range = [-3, 3],
            v_range = [-3, 3],
            fill_opacity = 0.5,
            fill_color = ORANGE
        ).set_fill_by_checkerboard(RED, RED)

        celestial_circle = ParametricFunction(
            lambda t: np.array([np.cos(t), np.sin(t), 1]),
            t_range = [0, 2 * np.pi],
            color = PURPLE
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

        v = np.array([-1/3, 2/3])
        Lambda = vec_to_boost_2(v)

        # Animations

        self.add(light_cone_bottom, time_slice, celestial_circle, stars, light_cone_top)
        
        self.play(ApplyMatrix(matrix = Lambda, mobject = celestial_circle),
        ApplyMatrix(matrix = Lambda, mobject = light_cone_bottom),
        ApplyMatrix(matrix = Lambda, mobject = time_slice),
        ApplyMatrix(matrix = Lambda, mobject = light_cone_top),
        ApplyMatrix(matrix = Lambda, mobject = stars))
        self.play(ApplyPointwiseFunction(rescaling_2, celestial_circle),
        ApplyPointwiseFunction(rescaling_2, stars), 
        ApplyPointwiseFunction(lambda X: [X[0], X[1], 1], time_slice))

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

        self.add(celestial_sphere, stars)
        self.play(ApplyPointwiseFunction(lambda x: lorentz_transform_spacelike_proj(x, active_transformation), celestial_sphere),
                ApplyPointwiseFunction(lambda x: lorentz_transform_spacelike_proj(x, active_transformation), stars))
        self.play(ApplyPointwiseFunction(rescaling_3, celestial_sphere),
                ApplyPointwiseFunction(rescaling_3, stars))
        self.wait(1)

class ConformalTransformation1_3(ThreeDScene):
    def construct(self):

        # Camera

        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(3)

        # Mobjects
        celestial_sphere = Surface(
            lambda u, v: [np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)],
            resolution = (16, 16),
            u_range = [0, 2 * np.pi],
            v_range = [0, np.pi],
            fill_opacity = 0.5
        ).set_fill_by_checkerboard(YELLOW, GREEN)

        stars = Mobject1D()
        star_positions = []
        n_U = 8
        n_V = 8
        for n in range(n_U):
            for m in range(n_V):
                U = n / n_U
                V = m / n_V
                u = 2 * np.pi * U
                v = np.arccos(1 - 2 * V)
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

        v = [0, 0, -1]
        Lambda = vec_to_boost_3(v)

        # Animations

        self.add(celestial_sphere, stars)
        self.play(ApplyPointwiseFunction(lambda x: conformal_transformation_3_1(x, Lambda), celestial_sphere),
                ApplyPointwiseFunction(lambda x: conformal_transformation_3_1(x, Lambda), stars))
        self.wait(1)

class RigidRotation(ThreeDScene):
    def construct(self):

        # Camera

        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(3)

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
                v = np.arccos(1 - 2 * V)
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

        self.add(celestial_sphere, stars)
        self.play(Rotate(celestial_sphere, angle = np.pi/8, about_point = ORIGIN, rate_func =linear),
                Rotate(stars, angle = np.pi/8, about_point = ORIGIN, rate_func =linear))
        self.wait(1)

class MobiusTransformationCx(ThreeDScene):
    def construct(self):

        # Camera

        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(1.8)

        # Mobjects
        eps = 1e-1
        celestial_sphere = Surface(
            lambda u, v: [np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)],
            resolution = (16, 16),
            u_range = [0, 2 * np.pi],
            v_range = [eps, np.pi],
            fill_opacity = 0.5
        ).set_fill_by_checkerboard(PURPLE, PURPLE)

        stars = Mobject1D()
        star_positions = []
        n_U = 8
        n_V = 8
        for n in range(n_U):
            for m in range(n_V):
                U = n / n_U
                V = (m + 1) / n_V
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

        S = np.array([[np.exp(-0.4), 0],
                     [0, np.exp(0.4)]])

        # Animations

        self.add(celestial_sphere, stars)
        self.play(ApplyPointwiseFunction(stereographic_projection, celestial_sphere),
                ApplyPointwiseFunction(stereographic_projection, stars))
        self.play(ApplyPointwiseFunction(real_transformation_from_matrix(S), celestial_sphere),
                ApplyPointwiseFunction(real_transformation_from_matrix(S), stars))
        self.play(ApplyPointwiseFunction(stereographic_inverse, celestial_sphere),
                ApplyPointwiseFunction(stereographic_inverse, stars))
        self.wait(1)

class ConformalTransformationMobius(ThreeDScene):
    def construct(self):

        # Camera

        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(3)

        # Mobjects
        eps = 1e-1
        celestial_sphere = Surface(
            lambda u, v: [np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)],
            resolution = (16, 16),
            u_range = [0, 2 * np.pi],
            v_range = [eps, np.pi],
            fill_opacity = 0.5
        ).set_fill_by_checkerboard(YELLOW, GREEN)

        stars = Mobject1D()
        star_positions = []
        n_U = 16
        n_V = 16
        for n in range(n_U):
            for m in range(n_V):
                U = n / n_U
                V = (m + 1) / n_V
                u = 2 * np.pi * U
                v = np.arccos(1 - 2 * V)
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

        S = np.array([[np.exp(-1/2), 0],
                     [0, np.exp(1/2)]])

        # Animations

        self.add(celestial_sphere, stars)
        self.play(ApplyPointwiseFunction(lambda x: conformal_transformation_mobius(x, S), celestial_sphere),
                ApplyPointwiseFunction(lambda x: conformal_transformation_mobius(x, S), stars))
        self.wait(1)

class TransformationLoop(ThreeDScene):
    def construct(self):

        # Camera

        phi = 30
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(2)

        # Mobjects
        eps = 1e-1
        celestial_sphere = Surface(
            lambda u, v: [np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)],
            resolution = (16, 16),
            u_range = [0, 2 * np.pi],
            v_range = [eps, np.pi],
            fill_opacity = 0.5
        ).set_fill_by_checkerboard(YELLOW, GREEN)

        stars = Mobject1D()
        star_positions = []
        n_U = 16
        n_V = 16
        for n in range(n_U):
            for m in range(n_V):
                U = n / n_U
                V = (m + 1) / n_V
                u = 2 * np.pi * U
                v = np.arccos(1 - 2 * V)
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

        S_x = np.array([[np.cosh(1/2), np.sinh(-1/2)],
                     [np.sinh(-1/2), np.cosh(1/2)]])

        S_z = np.array([[np.exp(-1/2), 0],
                        [0, np.exp(1/2)]])
        S = S_z
        v = [+0, 0, +1]
        Lambda = vec_to_boost_3(v)

        # Animations

        self.add(celestial_sphere, stars)
        self.play(ApplyPointwiseFunction(stereographic_projection, celestial_sphere),
                ApplyPointwiseFunction(stereographic_projection, stars))
        self.play(ApplyPointwiseFunction(real_transformation_from_matrix(S), celestial_sphere),
                ApplyPointwiseFunction(real_transformation_from_matrix(S), stars))
        self.play(ApplyPointwiseFunction(stereographic_inverse, celestial_sphere),
                ApplyPointwiseFunction(stereographic_inverse, stars))
        self.play(ApplyPointwiseFunction(lambda x: lorentz_transform_spacelike_proj(x, Lambda), celestial_sphere),
                ApplyPointwiseFunction(lambda x: lorentz_transform_spacelike_proj(x, Lambda), stars))
        self.play(ApplyPointwiseFunction(rescaling_3, celestial_sphere),
                ApplyPointwiseFunction(rescaling_3, stars))
        self.wait(1)

class MobiusMaps(ThreeDScene):
    def construct(self):
        # Camera

        phi = 0
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(1)

        # Mobjects
        eps = 1e-1
        cx_plane = Surface(
            lambda u, v: [np.cos(u) * np.sin(v)/(1 - np.cos(v)), np.sin(u) * np.sin(v)/(1 - np.cos(v)), 0],
            resolution = (16, 16),
            u_range = [0, 2 * np.pi],
            v_range = [eps, np.pi],
            fill_opacity = 0.5
        ).set_fill_by_checkerboard(YELLOW, GREEN)

        stars = Mobject1D()
        star_positions = []
        n_U = 16
        n_V = 16
        for n in range(n_U):
            for m in range(n_V):
                U = n / n_U
                V = (m + 1) / n_V
                u = 2 * np.pi * U
                v = np.pi * V #np.arccos(1 - 2 * V)
                star_positions.append([u, v])
        stars.add_points(
            [
                [np.cos(position[0]) * np.sin(position[1])/(1 - np.cos(position[1])), 
                np.sin(position[0]) * np.sin(position[1])/(1 - np.cos(position[1])),
                0]
                for position in star_positions
            ]
        )

        # Transformations

        b_x = np.array([[0, 1],
                        [1, 0]])
        b_y = complex(0,1) * np.array([[0, -1],
                                      [1, 0]])
        b_z = np.array([[1, 0],
                        [0, -1]])
        B_x = expm(b_x/2)
        B_y = expm(b_y/2)
        B_z = expm(b_z/2)

        r_x = complex(0,1) * b_x
        r_y = complex(0,1) * b_y
        r_z = complex(0,1) * b_z

        R_x = expm(r_x/2 * np.pi/2)
        R_y = expm(r_y/2 * np.pi/2)
        R_z = expm(r_z/2 * np.pi/2)

        active_matrix = B_x

        # Animations
        self.play(ApplyPointwiseFunction(real_transformation_from_matrix(active_matrix), cx_plane),
        ApplyPointwiseFunction(real_transformation_from_matrix(active_matrix), stars),)
        self.wait(1)

        # Command
        # manim --quality=l -o<name of output file> lorentz.py MobiusMaps
        # manim --quality=l --flush_cache -ox_boost.mp4 lorentz.py MobiusMaps

class VFieldFlows(Scene):
    def construct(self):
        func_rot = lambda pos: - pos[1] * RIGHT + pos[0] * UP 
        func_dil = lambda pos: pos[0] * RIGHT + pos[1] * UP
        func_lox = lambda pos: (pos[0] - pos[1]) * RIGHT + (pos[0] + pos[1]) * UP
        func_x_rot = lambda pos: (2 * pos[0] * pos[1]) * RIGHT + (1 - pos[0]**2 + pos[1]**2) * UP

        func_boost = lambda pos: pos[1] * RIGHT + pos[0] * UP

        active_v_field = func_x_rot

        stream_lines = StreamLines(active_v_field,
                                   x_range=[-3,3,1],
                                   y_range=[-2,2, 1],
                                   color=YELLOW,
                                   stroke_width=3, 
                                   max_anchors_per_line=10,
                                   virtual_time=1)
        # Continuous motion
        """ self.add(stream_lines)
        stream_lines.start_animation(warm_up=True, flow_speed=1)
        self.wait(5) """

        # Streamline creation
        
        self.play(stream_lines.create())
        self.wait()
        

        