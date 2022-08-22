###
# mobius_maps.py
#
# scenes for mobius maps part of video
#
###

'''
Dependencies
'''

from manim import *
from lorentz_utils import *

'''
Scenes
'''

class ZRotateMobius(ThreeDScene):
    def construct(self):

        # Camera

        phi = 75
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(2)

        # Mobjects

        punctured_sphere = Surface(lambda u,v: np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)]),
                               u_range = [0, 2 * np.pi],
                               v_range = [np.pi/12, np.pi],
                               resolution = (12, 11),
                               fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE, PURPLE)

        stars = Mobject1D()
        n_phis = 6
        n_thetas = 12
        phis = np.linspace(0, 2 * np.pi, n_phis, endpoint=False)
        thetas = np.linspace(np.pi, 0, n_thetas, endpoint=False)
        
        for phi in phis:
            for theta in thetas:
                x = np.cos(phi) * np.sin(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(theta)
                stars.add_points([np.array([x, y, z])])

        self.play(Create(punctured_sphere))
        self.add(stars)
        self.wait()

        self.play(ApplyPointwiseFunction(stereographic_projection, punctured_sphere),
                  ApplyPointwiseFunction(stereographic_projection, stars))
        self.wait()

        self.play(Rotate(punctured_sphere, np.pi/6),
                  Rotate(stars, np.pi/6))
        self.wait()

        self.play(ApplyPointwiseFunction(stereographic_inverse, punctured_sphere),
                  ApplyPointwiseFunction(stereographic_inverse, stars))
        self.wait()

        self.play(Rotate(punctured_sphere, -np.pi/6),
                  Rotate(stars, -np.pi/6))
        self.wait()

        self.remove(stars)
        self.play(FadeOut(punctured_sphere))

class ZRotateMobiusDetailed(ThreeDScene):
    def construct(self):

        # Camera

        phi = 75
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(1.5)

        # Mobjects

        axes = ThreeDAxes(x_range = [0,1,1], y_range = [0,1,1], z_range = [0,1,1], x_length=1, y_length = 1, z_length = 1, z_normal = [0,+1,0]).set_x(0.42).set_y(0.42).set_z(0.5)
        z_label = axes.get_z_axis_label(MathTex('z'))

        re_label = axes.get_x_axis_label(MathTex(r'\text{Re}(w)'))
        im_label = axes.get_y_axis_label(MathTex(r'\text{Im}(w)'))

        punctured_sphere = Surface(lambda u,v: np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)]),
                               u_range = [0, 2 * np.pi],
                               v_range = [np.pi/12, np.pi],
                               resolution = (12, 11),
                               fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE, PURPLE)

        stars = Mobject1D()
        n_phis = 6
        n_thetas = 6
        phis = np.linspace(0, 2 * np.pi, n_phis, endpoint=False)
        thetas = np.linspace(np.pi, 0, n_thetas, endpoint=False)
        
        for phi in phis:
            for theta in thetas:
                x = np.cos(phi) * np.sin(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(theta)
                stars.add_points([np.array([x, y, z])])

        map_text = Tex(r'$w \mapsto e^{i\theta}w$').set_y(-2)

        # Animations

        self.play(FadeIn(axes, z_label))
        self.wait()

        self.play(FadeIn(punctured_sphere))
        self.add(stars)
        self.wait()

        self.play(ApplyPointwiseFunction(stereographic_projection, punctured_sphere),
                  ApplyPointwiseFunction(stereographic_projection, stars))
        self.wait()

        self.begin_ambient_camera_rotation(rate = -75 / 2 * DEGREES, about = 'phi')
        self.wait(2)

        self.stop_ambient_camera_rotation(about = 'phi')
        self.wait()

        self.play(FadeIn(re_label, im_label))
        self.wait()

        self.play(Write(map_text))
        self.wait()

        self.play(Rotate(punctured_sphere, np.pi/6),
                  Rotate(stars, np.pi/6))
        self.wait()

        self.play(Unwrite(map_text))
        self.wait()

        self.play(FadeOut(re_label, im_label))
        self.wait()

        self.begin_ambient_camera_rotation(rate = 75 / 2 * DEGREES, about = 'phi')
        self.wait(2)

        self.stop_ambient_camera_rotation(about = 'phi')
        self.wait()

        self.play(ApplyPointwiseFunction(stereographic_inverse, punctured_sphere),
                  ApplyPointwiseFunction(stereographic_inverse, stars))
        self.wait()

        self.play(Rotate(punctured_sphere, -np.pi/6),
                  Rotate(stars, -np.pi/6))
        self.wait()

        self.remove(stars)
        self.play(FadeOut(punctured_sphere, axes, z_label))

class ZBoostMobius(ThreeDScene):
    def construct(self):

        # Camera

        phi = 75
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(1.5)

        # Mobjects

        axes = ThreeDAxes(x_range = [0,1,1], y_range = [0,1,1], z_range = [0,1,1], x_length=1, y_length = 1, z_length = 1, z_normal = [0,+1,0]).set_x(0.42).set_y(0.42).set_z(0.5)
        z_label = axes.get_z_axis_label(MathTex('z'))

        punctured_sphere = Surface(lambda u,v: np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)]),
                               u_range = [0, 2 * np.pi],
                               v_range = [np.pi/12, np.pi],
                               resolution = (12, 11),
                               fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE, PURPLE)

        stars = Mobject1D()
        n_phis = 6
        n_thetas = 12
        phis = np.linspace(0, 2 * np.pi, n_phis, endpoint=False)
        thetas = np.linspace(np.pi, 0, n_thetas, endpoint=False)
        
        for phi in phis:
            for theta in thetas:
                x = np.cos(phi) * np.sin(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(theta)
                stars.add_points([np.array([x, y, z])])

        map_text = Tex(r'$w \mapsto \alpha w$').set_y(-2)

        # Animations

            # Create stuff

        self.play(FadeIn(axes, z_label, punctured_sphere))
        self.add(stars)
        self.wait()

            # Stereographic projection

        self.play(ApplyPointwiseFunction(stereographic_projection, punctured_sphere),
                  ApplyPointwiseFunction(stereographic_projection, stars))
        self.wait()

        self.begin_ambient_camera_rotation(rate = -75 / 2 * DEGREES, about = 'phi')
        self.wait(2)

        self.stop_ambient_camera_rotation(about = 'phi')
        self.wait()

            # Complex transformation

        self.play(Write(map_text))
        self.wait()

        self.play(ApplyComplexFunction(lambda z: np.exp(-.8) * z, punctured_sphere),
                  ApplyComplexFunction(lambda z: np.exp(-.8) * z, stars))
        self.wait()

        self.play(Unwrite(map_text))
        self.wait()

        self.begin_ambient_camera_rotation(rate = 75 / 2 * DEGREES, about = 'phi')
        self.wait(2)

        self.stop_ambient_camera_rotation(about = 'phi')
        self.wait()

            # Undo projection

        self.play(ApplyPointwiseFunction(stereographic_inverse, punctured_sphere),
                  ApplyPointwiseFunction(stereographic_inverse, stars))
        self.wait()

        self.play(ApplyPointwiseFunction(lambda x: conformal_transformation_3_1(x, Lambda = SL_to_SO(np.array([[np.exp(.4),0],[0, np.exp(-.4)]]))), punctured_sphere),
                  ApplyPointwiseFunction(lambda x: conformal_transformation_3_1(x, Lambda = SL_to_SO(np.array([[np.exp(.4),0],[0, np.exp(-.4)]]))), stars))
        self.wait()

        self.remove(stars)
        self.play(FadeOut(punctured_sphere, axes, z_label))

class InversionMobius(ThreeDScene):
    def construct(self):

        # Camera

        phi = 75
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(1.5)

        # Mobjects

        axes = ThreeDAxes(x_range = [0,1,1], y_range = [0,1,1], z_range = [0,1,1], x_length=1, y_length = 1, z_length = 1, z_normal = [0,+1,0]).set_x(0.42).set_y(0.42).set_z(0.5)
        z_label = axes.get_z_axis_label(MathTex('z'))

        punctured_sphere = Surface(lambda u,v: np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)]),
                               u_range = [0, 2 * np.pi],
                               v_range = [np.pi/12, np.pi-np.pi/12],
                               resolution = (12, 10),
                               fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE, PURPLE)

        stars = Mobject1D()
        n_phis = 6
        n_thetas = 11
        phis = np.linspace(0, 2 * np.pi, n_phis, endpoint=False)
        thetas = np.linspace(np.pi-np.pi/12, 0, n_thetas, endpoint=False)
        
        for phi in phis:
            for theta in thetas:
                x = np.cos(phi) * np.sin(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(theta)
                stars.add_points([np.array([x, y, z])])

        map_text = Tex(r'$w \mapsto \frac{1}{w}$').set_y(-2)

        # Animations

            # Create stuff

        self.play(FadeIn(axes, z_label, punctured_sphere))
        self.add(stars)
        self.wait()

            # Stereographic projection

        self.play(ApplyPointwiseFunction(stereographic_projection, punctured_sphere),
                  ApplyPointwiseFunction(stereographic_projection, stars))
        self.wait()

        self.begin_ambient_camera_rotation(rate = -75 / 2 * DEGREES, about = 'phi')
        self.wait(2)

        self.stop_ambient_camera_rotation(about = 'phi')
        self.wait()

            # Complex tfmn

        self.play(Write(map_text))
        self.wait()

        self.play(ApplyPointwiseFunction(lambda x: x/np.linalg.norm(x)**2, punctured_sphere, run_time = 2),
                  ApplyPointwiseFunction(lambda x: x/np.linalg.norm(x)**2, stars, run_time = 2))
        self.wait()

        self.play(ApplyPointwiseFunction(lambda x: np.array([x[0], -x[1], 0]), punctured_sphere, run_time = 2),
                  ApplyPointwiseFunction(lambda x: np.array([x[0], -x[1], 0]), stars, run_time = 2))
        self.wait()

        self.play(Unwrite(map_text))
        self.wait()

        self.begin_ambient_camera_rotation(rate = 75 / 2 * DEGREES, about = 'phi')
        self.wait(2)

        self.stop_ambient_camera_rotation(about = 'phi')
        self.wait()

            # Undo projection

        self.play(ApplyPointwiseFunction(stereographic_inverse, punctured_sphere),
                  ApplyPointwiseFunction(stereographic_inverse, stars))
        self.wait()

        self.play(Rotate(punctured_sphere, np.pi, axis = np.array([1, 0, 0])),
                  Rotate(stars, np.pi, axis = np.array([1, 0, 0])))
        self.wait()

        self.remove(stars)
        self.play(FadeOut(punctured_sphere, axes, z_label))

class StereographicMap(ThreeDScene):
    def construct(self):
        
        # Camera

        phi = 75
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(1.5)

        axes = ThreeDAxes(x_range = [0,1,1], y_range = [0,1,1], z_range = [0,1,1], x_length=1, y_length = 1, z_length = 1, z_normal = [0,+1,0]).set_x(0.42).set_y(0.42).set_z(0.5)
        z_label = axes.get_z_axis_label(MathTex('N = (0,0,1)'))

        sphere = Surface(lambda u,v: np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)]),
                         u_range = [0, 2 * np.pi],
                         v_range = [0, np.pi],
                         resolution = (12, 12),
                         fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE, PURPLE)

        lower_sphere = Surface(lambda u,v: np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)]),
                         u_range = [0, 2 * np.pi],
                         v_range = [np.pi/2, np.pi],
                         resolution = (12, 6),
                         fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE_A, PURPLE_A)

        upper_sphere = Surface(lambda u,v: np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)]),
                         u_range = [0, 2 * np.pi],
                         v_range = [0, np.pi/2],
                         resolution = (12, 6),
                         fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE_C, PURPLE_C)

        punctured_sphere = Surface(lambda u,v: np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)]),
                               u_range = [0, 2 * np.pi],
                               v_range = [np.pi/12, np.pi],
                               resolution = (12, 11),
                               fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE, PURPLE)

        plane_range = 5
        plane = Surface(lambda u,v: np.array([u, v, 0]),
                        u_range = [-plane_range, +plane_range],
                        v_range = [-plane_range, +plane_range],
                        resolution = (1, 1),
                        fill_opacity= 0.7).set_fill_by_checkerboard(ORANGE, ORANGE)

        inner_lines = list()
        outer_lines = list()

        phi = -np.pi/6
        thetas = [np.pi/3, np.pi/6]
        start_point = np.array([0,0,1])
        for theta in thetas:
            x = np.cos(phi) * np.sin(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(theta)
            
            mid_point = np.array([x,y,z])
            end_point = np.array([x/(1-z), y/(1-z), 0])

            inner_line = Line(start_point, mid_point, color = RED_B)
            outer_line = Line(mid_point, end_point, color = RED_A)
            inner_lines.append(inner_line)
            outer_lines.append(outer_line)
        inner_lines.append(Line(start_point, start_point, color = RED_B))
        outer_lines.append(Line(start_point, np.array([x * 50, y * 50, 1]), color = RED_A))

        north_pole = Dot3D(point = np.array([0,0,1]))

        equator = ParametricFunction(lambda t: np.array([np.cos(t), np.sin(t), 0]),
                                     t_range = [0, 2 * np.pi],
                                     color = ORANGE)

        # Animations

        self.play(FadeIn(sphere, axes))
        self.wait()

        self.play(FadeIn(north_pole, z_label))
        self.wait()

        self.play(FadeOut(sphere, axes, z_label, north_pole), FadeIn(upper_sphere, lower_sphere, plane))
        self.wait()

        self.play(Create(inner_lines[0]))
        self.wait()

        self.play(Create(outer_lines[0]))
        self.wait()

        self.play(Transform(inner_lines[0], inner_lines[1]),
                  Transform(outer_lines[0], outer_lines[1]))
        self.wait()

        self.play(Transform(inner_lines[0], inner_lines[2]),
                  Transform(outer_lines[0], outer_lines[2]))
        self.wait()

        self.play(FadeOut(upper_sphere, lower_sphere, plane, inner_lines[0], outer_lines[0]), FadeIn(punctured_sphere))
        self.wait()

        self.play(ApplyPointwiseFunction(stereographic_projection, punctured_sphere))
        self.wait()

        self.play(FadeIn(equator))
        self.wait()

        self.play(FadeOut(equator))
        self.wait()

        self.play(ApplyPointwiseFunction(stereographic_inverse, punctured_sphere))
        self.wait()

        self.play(FadeOut(punctured_sphere))
        self.wait()

class LabelledAxes(ThreeDScene):
    def construct(self):
        # Camera

        phi = 0
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        # Mobjects 

        axes = ThreeDAxes(x_range = [0,1,1], y_range = [0,1,1], z_range = [0,1,1], x_length=1, y_length = 1, z_length = 1, z_normal = [0,+1,0])
        x_label = axes.get_x_axis_label(Tex('x'))
        y_label = axes.get_y_axis_label(Tex('y'))
        z_label = axes.get_z_axis_label(Tex('z'))

        # Animations

        self.play(FadeIn(axes, x_label, y_label, z_label))
        self.wait()

        self.begin_ambient_camera_rotation(rate = 75/2*DEGREES, about = 'phi')
        self.wait(2)
        self.stop_ambient_camera_rotation(about = 'phi')
        self.wait()

        self.play(FadeOut(axes, x_label, y_label, z_label))
        self.wait()

class InversionInfinity(ThreeDScene):
    def construct(self):
        # Camera

        phi = 75
        theta = -90
        self.set_camera_orientation(phi = phi * DEGREES, theta = theta * DEGREES)
        self.camera.set_zoom(1.5)

        # Function 

        def inversion_3D(X):
            '''
            Parameters
            ----------
            X (np.array): 3d array
            
            Returns
            -------
            Y (np.array): image of inversion / rotation by pi around x axis
            '''
            x, y, z = X
            Y = np.array([x, -y, -z])
            return Y

        # Mobjects

        axes = ThreeDAxes(x_range = [0,1,1], y_range = [0,1,1], z_range = [0,1,1], x_length=1, y_length = 1, z_length = 1, z_normal = [0,+1,0]).set_x(0.42).set_y(0.42).set_z(0.5)

        punctured_sphere = Surface(lambda u,v: np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)]),
                               u_range = [0, 2 * np.pi],
                               v_range = [np.pi/12, np.pi-np.pi/12],
                               resolution = (12, 10),
                               fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE, PURPLE)

        sphere = Surface(lambda u,v: np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)]),
                         u_range = [0, 2 * np.pi],
                         v_range = [0, np.pi],
                         resolution = (12, 12),
                         fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE, PURPLE)

            # Stars (for punctured sphere)
        punctured_stars = Mobject1D()
        n_phis = 6
        n_thetas = 11
        phis = np.linspace(0, 2 * np.pi, n_phis, endpoint=False)
        thetas = np.linspace(np.pi-np.pi/12, 0, n_thetas, endpoint=False)
        
        for phi in phis:
            for theta in thetas:
                x = np.cos(phi) * np.sin(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(theta)
                punctured_stars.add_points([np.array([x, y, z])])

            # Stars (for full sphere)
        stars = Mobject1D()
        n_phis = 6
        n_thetas = 13
        phis = np.linspace(0, 2 * np.pi, n_phis, endpoint=False)
        thetas = np.linspace(np.pi, 0, n_thetas, endpoint=True)
        
        for phi in phis:
            for theta in thetas:
                x = np.cos(phi) * np.sin(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(theta)
                stars.add_points([np.array([x, y, z])])

        # Animations

        self.play(FadeIn(punctured_sphere, axes))
        self.add(punctured_stars)
        self.wait()

        self.play(ApplyPointwiseFunction(inversion_3D, punctured_sphere, run_time=2), ApplyPointwiseFunction(inversion_3D, punctured_stars, run_time=2))
        self.wait()

        self.begin_ambient_camera_rotation(rate = np.pi/2, about = 'theta')
        self.wait()
        self.stop_ambient_camera_rotation(about = 'theta')
        self.wait()

        self.play(ApplyPointwiseFunction(inversion_3D, punctured_sphere, run_time=2), ApplyPointwiseFunction(inversion_3D, punctured_stars, run_time=2))
        self.wait()

        self.begin_ambient_camera_rotation(rate = -np.pi/2, about = 'theta')
        self.wait()
        self.stop_ambient_camera_rotation(about = 'theta')
        self.wait()

        self.play(FadeOut(punctured_sphere), FadeIn(sphere))
        self.add(stars)
        self.remove(punctured_stars)
        self.wait()

        self.play(ApplyPointwiseFunction(inversion_3D, sphere, run_time=2), ApplyPointwiseFunction(inversion_3D, stars, run_time=2))
        self.wait()

        self.remove(stars)
        self.play(FadeOut(sphere, axes))

class DifferentRotations(ThreeDScene):
    def construct(self):

        # Camera

        phi = 75
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(2)

        # Mobjects

        sphere = Surface(lambda u,v: np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)]),
                         u_range = [0, 2 * np.pi],
                         v_range = [0, np.pi],
                         resolution = (12, 12),
                         fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE, PURPLE)

        # Animations

        self.play(FadeIn(sphere))
        self.wait()

        self.play(Rotate(sphere, np.pi/2))
        self.wait()

        self.play(Rotate(sphere, np.pi/2, axis = np.array([1,0,0])))
        self.wait()

        self.play(Rotate(sphere, np.pi/2, axis = np.array([0,1,0])))
        self.wait()

        self.play(FadeOut(sphere))

class IsolatedRotations(ThreeDScene):
    def construct(self):

        # Camera

        phi = 75
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(1.5)

        # Mobjects

        sphere = Surface(lambda u,v: np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)]),
                         u_range = [0, 2 * np.pi],
                         v_range = [0, np.pi],
                         resolution = (12, 12),
                         fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE, PURPLE)

        stars = Mobject1D()
        n_phis = 6
        n_thetas = 13
        phis = np.linspace(0, 2 * np.pi, n_phis, endpoint=False)
        thetas = np.linspace(np.pi, 0, n_thetas, endpoint=True)
        
        for phi in phis:
            for theta in thetas:
                x = np.cos(phi) * np.sin(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(theta)
                stars.add_points([np.array([x, y, z])])

        # Animations

        self.play(FadeIn(sphere))
        self.add(stars)
        self.wait()

        self.play(Rotate(sphere, np.pi/3), Rotate(stars, np.pi/3))
        self.wait()

        self.remove(stars)
        self.play(FadeOut(sphere))
        self.wait()

        self.play(FadeIn(sphere))
        self.add(stars)
        self.wait()

        self.play(Rotate(sphere, np.pi/2, axis = np.array([1,0,0])),
                  Rotate(stars, np.pi/2, axis = np.array([1,0,0])))
        self.wait()

        self.remove(stars)
        self.play(FadeOut(sphere))
        self.wait()

class StereoNetMobiusMap(ThreeDScene):
    def construct(self):

        # Camera

        phi = 0
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(1.5)

        # Mobjects

        stereo_net = Surface(lambda u, v: np.array([np.cos(u) * np.sin(v)/(1 - np.cos(v)), np.sin(u) * np.sin(v)/(1 - np.cos(v)), 0]),
                             u_range = [0, 2 * np.pi],
                             v_range = [np.pi/24, np.pi - np.pi/24],
                             resolution = (12, 11),
                             fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE, PURPLE)

        stars = Mobject1D()
        n_phis = 12
        n_thetas = 11
        phis = np.linspace(0, 2 * np.pi, n_phis, endpoint=False)
        thetas = np.linspace(np.pi/24, np.pi - np.pi/24, n_thetas, endpoint=False)
        for phi in phis:
            for theta in thetas:
                x = np.cos(phi) * np.sin(theta)/(1 - np.cos(theta))
                y = np.sin(phi) * np.sin(theta)/(1 - np.cos(theta))
                z = 0
                stars.add_points([np.array([x, y, z])])

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

        active_matrix = R_x

        # Animations
        self.play(FadeIn(stereo_net))
        self.add(stars)
        self.wait()
        
        self.play(ApplyPointwiseFunction(real_transformation_from_matrix(active_matrix), stereo_net),
        ApplyPointwiseFunction(real_transformation_from_matrix(active_matrix), stars))
        self.wait()

        self.remove(stars)
        self.play(FadeOut(stereo_net))

class StereoNetRotation(ThreeDScene):
    def construct(self):

        # Camera

        phi = 0
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(1.5)

        # Mobjects

        stereo_net = Surface(lambda u, v: np.array([np.cos(u) * np.sin(v)/(1 - np.cos(v)), np.sin(u) * np.sin(v)/(1 - np.cos(v)), 0]),
                             u_range = [0, 2 * np.pi],
                             v_range = [np.pi/12, np.pi],
                             resolution = (12, 11),
                             fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE, PURPLE)

        stars = Mobject1D()
        n_phis = 12
        n_thetas = 11
        phis = np.linspace(0, 2 * np.pi, n_phis, endpoint=False)
        thetas = np.linspace(np.pi/12, np.pi, n_thetas, endpoint=False)
        for phi in phis:
            for theta in thetas:
                x = np.cos(phi) * np.sin(theta)/(1 - np.cos(theta))
                y = np.sin(phi) * np.sin(theta)/(1 - np.cos(theta))
                z = 0
                stars.add_points([np.array([x, y, z])])

        map_text = Tex('$w \mapsto f(w)$').set_y(-2)

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

        active_matrix = R_z

        # Animations
        self.play(FadeIn(stereo_net))
        self.add(stars)
        self.wait()

        self.play(Write(map_text))
        self.wait()

        self.play(ApplyComplexFunction(map_from_matrix(-B_z - 2/3 * R_z), stereo_net),
                  ApplyComplexFunction(map_from_matrix(-B_z - 2/3 * R_z), stars))
        self.wait()

        self.remove(stars)
        self.play(Unwrite(map_text), FadeOut(stereo_net))

class ZRotOrbitsSphere(ThreeDScene):
    def construct(self):

        # Camera

        phi = 75
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(2)

        # Mobjects

        sphere = Surface(lambda u,v: np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)]),
                         u_range = [0, 2 * np.pi],
                         v_range = [0, np.pi],
                         resolution = (12, 12),
                         fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE, PURPLE)

        full_stars = Mobject1D()
        n_phis = 6
        n_thetas = 13
        phis = np.linspace(0, 2 * np.pi, n_phis, endpoint=False)
        thetas = np.linspace(np.pi, 0, n_thetas, endpoint=True)
        for phi in phis:
            for theta in thetas:
                x = np.cos(phi) * np.sin(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(theta)
                full_stars.add_points([np.array([x, y, z])])

        restricted_stars = Mobject1D()
        n_phis = 1
        n_thetas = 13
        phis = np.linspace(0, 2 * np.pi, n_phis, endpoint=False)
        thetas = np.linspace(np.pi, 0, n_thetas, endpoint=True)
        for phi in phis:
            for theta in thetas:
                x = np.cos(phi) * np.sin(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(theta)
                restricted_stars.add_points([np.array([x, y, z])])

        orbits = []
            # Reuse thetas from stars
        for theta in thetas:
            orbit = ParametricFunction(
                lambda t: np.array([np.cos(t) * np.sin(theta), np.sin(t) * np.sin(theta), np.cos(theta)]),
                t_range = [0, 2 * np.pi],
            )
            orbits.append(orbit)

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
                                color = interpolate_color(BLUE_D, YELLOW, (basepoint[2] + 1)/2)
                                ).set_x(vec_base[0]).set_y(vec_base[1]).set_z(vec_base[2])
                vector_field.add(vector)

        # Animations

        self.play(FadeIn(sphere))
        self.add(full_stars, restricted_stars)
        self.wait()

        self.remove(full_stars)
        self.wait()

        self.play(
            Rotate(sphere, 2 * np.pi), 
            Rotate(restricted_stars, 2 * np.pi, about_point = ORIGIN), 
            Create(orbits[0]),
            Create(orbits[1]),
            Create(orbits[2]),
            Create(orbits[3]),
            Create(orbits[4]),
            Create(orbits[5]),
            Create(orbits[6]),
            Create(orbits[7]),
            Create(orbits[8]),
            Create(orbits[9]),
            Create(orbits[10]),
            Create(orbits[11]),
            Create(orbits[12]),
            )
        self.wait()

        self.remove(restricted_stars)
        self.play(FadeOut(*orbits), FadeIn(vector_field))
        self.wait()

        self.play(
            Rotate(sphere, np.pi/3), 
            Rotate(vector_field, np.pi/3))
        self.wait()

        self.play(FadeOut(sphere, vector_field))

class ZRotOrbitsSphereChangingTheta(ThreeDScene):
    def construct(self):

        # Camera

        phi = 75
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(2)

        # Mobjects

        sphere = Surface(lambda u,v: np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)]),
                         u_range = [0, 2 * np.pi],
                         v_range = [0, np.pi],
                         resolution = (12, 12),
                         fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE, PURPLE)

        full_stars = Mobject1D()
        n_phis = 6
        n_thetas = 13
        phis = np.linspace(0, 2 * np.pi, n_phis, endpoint=False)
        thetas = np.linspace(np.pi, 0, n_thetas, endpoint=True)
        for phi in phis:
            for theta in thetas:
                x = np.cos(phi) * np.sin(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(theta)
                full_stars.add_points([np.array([x, y, z])])

        restricted_stars = Mobject1D()
        n_phis = 1
        n_thetas = 13
        phis = np.linspace(0, 2 * np.pi, n_phis, endpoint=False)
        thetas = np.linspace(np.pi, 0, n_thetas, endpoint=True)
        for phi in phis:
            for theta in thetas:
                x = np.cos(phi) * np.sin(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(theta)
                restricted_stars.add_points([np.array([x, y, z])])

        orbits_1 = []
        orbits_2 = []
        orbits_3 = []
            # Reuse thetas from stars
        for theta in thetas:
            upper_phis = np.arange(2 * np.pi/3, 2 * np.pi + 2 * np.pi/3, 2 * np.pi/3)
            orbit_1 = ParametricFunction(
                lambda t: np.array([np.cos(t) * np.sin(theta), np.sin(t) * np.sin(theta), np.cos(theta)]),
                t_range = [0, upper_phis[0]],
            )
            orbits_1.append(orbit_1)

            orbit_2 = ParametricFunction(
                lambda t: np.array([np.cos(t) * np.sin(theta), np.sin(t) * np.sin(theta), np.cos(theta)]),
                t_range = [upper_phis[0], upper_phis[1]],
            )
            orbits_2.append(orbit_2)

            orbit_3 = ParametricFunction(
                lambda t: np.array([np.cos(t) * np.sin(theta), np.sin(t) * np.sin(theta), np.cos(theta)]),
                t_range = [upper_phis[1], upper_phis[2]],
            )
            orbits_3.append(orbit_3)


        # Animations

        self.play(FadeIn(sphere))
        self.add(full_stars, restricted_stars)
        self.wait()

        self.remove(full_stars)
        self.wait()

        self.play(
            Rotate(sphere, 2 * np.pi/3), 
            Rotate(restricted_stars, 2 * np.pi/3, about_point = ORIGIN), 
            Create(orbits_1[0]),
            Create(orbits_1[1]),
            Create(orbits_1[2]),
            Create(orbits_1[3]),
            Create(orbits_1[4]),
            Create(orbits_1[5]),
            Create(orbits_1[6]),
            Create(orbits_1[7]),
            Create(orbits_1[8]),
            Create(orbits_1[9]),
            Create(orbits_1[10]),
            Create(orbits_1[11]),
            Create(orbits_1[12]),
            )
        self.wait()
        self.play(
            Rotate(sphere, 2 * np.pi/3), 
            Rotate(restricted_stars, 2 * np.pi/3, about_point = ORIGIN),
            Create(orbits_2[0]),
            Create(orbits_2[1]),
            Create(orbits_2[2]),
            Create(orbits_2[3]),
            Create(orbits_2[4]),
            Create(orbits_2[5]),
            Create(orbits_2[6]),
            Create(orbits_2[7]),
            Create(orbits_2[8]),
            Create(orbits_2[9]),
            Create(orbits_2[10]),
            Create(orbits_2[11]),
            Create(orbits_2[12]),
            )
        self.wait()

        self.play(
            Rotate(sphere, 2 * np.pi/3), 
            Rotate(restricted_stars, 2 * np.pi/3, about_point = ORIGIN),
            Create(orbits_3[0]),
            Create(orbits_3[1]),
            Create(orbits_3[2]),
            Create(orbits_3[3]),
            Create(orbits_3[4]),
            Create(orbits_3[5]),
            Create(orbits_3[6]),
            Create(orbits_3[7]),
            Create(orbits_3[8]),
            Create(orbits_3[9]),
            Create(orbits_3[10]),
            Create(orbits_3[11]),
            Create(orbits_3[12]),
            )
        self.wait()

        self.remove(restricted_stars)
        self.play(FadeOut(sphere, *orbits_1, *orbits_2, *orbits_3))

class ZRotOrbitsCxPlane(ThreeDScene):
    def construct(self):

        # Camera

        phi = 0
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(2)

        # Mobjects

        stereo_net = Surface(lambda u,v: np.array([np.cos(u) * (np.sin(v)/(1-np.cos(v))), np.sin(u) * (np.sin(v)/(1-np.cos(v))), 0]),
                         u_range = [0, 2 * np.pi],
                         v_range = [np.pi/12, np.pi],
                         resolution = (12, 11),
                         fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE, PURPLE)

        full_stars = Mobject1D()
        n_phis = 6
        n_thetas = 12
        phis = np.linspace(0, 2 * np.pi, n_phis, endpoint=False)
        thetas = np.linspace(np.pi/12, np.pi, n_thetas, endpoint=True)
        for phi in phis:
            for theta in thetas:
                x = np.cos(phi) * (np.sin(theta)/(1 - np.cos(theta)))
                y = np.sin(phi) * (np.sin(theta)/(1 - np.cos(theta)))
                z = 0
                full_stars.add_points([np.array([x, y, z])])

        restricted_stars = Mobject1D()
        n_phis = 1
        n_thetas = 12
        phis = np.linspace(0, 2 * np.pi, n_phis, endpoint=False)
        thetas = np.linspace(np.pi/12, np.pi, n_thetas, endpoint=True)
        for phi in phis:
            for theta in thetas:
                x = np.cos(phi) * (np.sin(theta)/(1 - np.cos(theta)))
                y = np.sin(phi) * (np.sin(theta)/(1 - np.cos(theta)))
                z = 0
                restricted_stars.add_points([np.array([x, y, z])])

        orbits = []
            # Reuse thetas from stars
        for theta in thetas:
            orbit = ParametricFunction(
                lambda t: np.array([np.cos(t) * (np.sin(theta)/(1 - np.cos(theta))), np.sin(t) * (np.sin(theta)/(1 - np.cos(theta))), 0]),
                t_range = [0, 2 * np.pi],
            )
            orbits.append(orbit)

        vector_field = VGroup()
        phis = np.arange(0, 2 * np.pi, np.pi/6)
        for phi in phis:
            for theta in thetas:
                x = np.cos(phi) * (np.sin(theta)/(1 - np.cos(theta)))
                y = np.sin(phi) * (np.sin(theta)/(1 - np.cos(theta)))
                z = 0

                basepoint = np.array([x,y,z])
                vec_array = np.array([-y, x, 0])/3
                vec_base = basepoint + 0.5 * vec_array
                vector = Vector(vec_array,
                                color = interpolate_color(BLUE_D, YELLOW, (np.cos(theta) + 1)/2)
                                ).set_x(vec_base[0]).set_y(vec_base[1]).set_z(vec_base[2])
                vector_field.add(vector)

        # Animations

        self.play(FadeIn(stereo_net))
        self.add(full_stars, restricted_stars)
        self.wait()

        self.remove(full_stars)
        self.wait()

        self.play(
            Rotate(stereo_net, 2 * np.pi), 
            Rotate(restricted_stars, 2 * np.pi, about_point = ORIGIN), 
            Create(orbits[0]),
            Create(orbits[1]),
            Create(orbits[2]),
            Create(orbits[3]),
            Create(orbits[4]),
            Create(orbits[5]),
            Create(orbits[6]),
            Create(orbits[7]),
            Create(orbits[8]),
            Create(orbits[9]),
            Create(orbits[10]),
            Create(orbits[11]),
            )
        self.wait()

        self.remove(restricted_stars)
        self.play(FadeOut(*orbits), FadeIn(vector_field))
        self.wait()

        self.play(
            Rotate(stereo_net, np.pi/3), 
            Rotate(vector_field, np.pi/3),
            )
        self.wait()

        self.play(FadeOut(stereo_net, vector_field))

class ZBoostOrbitsSphere(ThreeDScene):
    def construct(self):

        # Camera

        phi = 75
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(2)

        # Mobjects

        sphere = Surface(lambda u,v: np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)]),
                         u_range = [0, 2 * np.pi],
                         v_range = [0, np.pi],
                         resolution = (12, 12),
                         fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE, PURPLE)

        orbits = []
        phis = np.arange(0, 2 * np.pi, np.pi/3)
        thetas = np.arange(0, np.pi, np.pi/12)
        for phi in phis:
            orbit = ParametricFunction(
                lambda t: np.array([np.cos(phi) * np.sin(t), np.sin(phi) * np.sin(t), np.cos(t)]),
                t_range = [0, np.pi],
            )
            orbits.append(orbit)

        vector_field = VGroup()
        for phi in phis:
            for theta in thetas:
                x = np.cos(phi) * np.sin(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(theta)

                basepoint = np.array([x,y,z])
                vec_array = np.array([-np.cos(phi) * np.cos(theta), -np.sin(phi) * np.cos(theta), +np.sin(theta)])/3
                vec_base = basepoint - 0.5 * vec_array
                vector = Vector(vec_array,
                                color = interpolate_color(BLUE_D, YELLOW, (basepoint[2] + 1)/2)
                                ).set_x(vec_base[0]).set_y(vec_base[1]).set_z(vec_base[2])
                vector_field.add(vector)

        # Animations

        self.play(FadeIn(sphere))
        self.wait()

        self.wait()

        self.play(
            Create(orbits[0]),
            Create(orbits[1]),
            Create(orbits[2]),
            Create(orbits[3]),
            Create(orbits[4]),
            Create(orbits[5]),
            )
        self.wait()
        self.play(FadeOut(*orbits), FadeIn(vector_field))
        self.wait()

        self.play(
            ApplyPointwiseFunction(lambda x: conformal_transformation_3_1(x, Lambda = SL_to_SO(np.array([[np.exp(.2),0],[0, np.exp(-.2)]]))), sphere),
            ApplyPointwiseFunction(lambda x: conformal_transformation_3_1(x, Lambda = SL_to_SO(np.array([[np.exp(.2),0],[0, np.exp(-.2)]]))), vector_field),)
        self.wait()

        self.play(FadeOut(sphere, vector_field))

class ZBoostOrbitsCxPlane(ThreeDScene):
    def construct(self):

        # Camera

        phi = 0
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(2)

        # Mobjects

        stereo_net = Surface(lambda u,v: np.array([np.cos(u) * (np.sin(v)/(1-np.cos(v))), np.sin(u) * (np.sin(v)/(1-np.cos(v))), 0]),
                         u_range = [0, 2 * np.pi],
                         v_range = [np.pi/12, np.pi],
                         resolution = (12, 11),
                         fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE, PURPLE)

        orbits = []
        phis = np.arange(0, 2 * np.pi, np.pi/3)
        thetas = np.arange(np.pi/12, np.pi, np.pi/12)
        for phi in phis:
            orbit = ParametricFunction(
                lambda t: np.array([np.cos(phi) * (np.sin(np.pi - t)/(1 - np.cos(np.pi - t))), np.sin(phi) * (np.sin(np.pi - t)/(1 - np.cos(np.pi - t))), 0]),
                t_range = [0, np.pi - np.pi/12],
            )
            orbits.append(orbit)

        vector_field = VGroup()
        for phi in phis:
            for theta in thetas:
                x = np.cos(phi) * (np.sin(theta)/(1 - np.cos(theta)))
                y = np.sin(phi) * (np.sin(theta)/(1 - np.cos(theta)))
                z = 0

                basepoint = np.array([x,y,z])
                vec_array = np.array([x, y, 0])/3
                vec_base = basepoint + 0.5 * vec_array
                vector = Vector(vec_array,
                                color = interpolate_color(BLUE_D, YELLOW, (np.cos(theta) + 1)/2)
                                ).set_x(vec_base[0]).set_y(vec_base[1]).set_z(vec_base[2])
                vector_field.add(vector)

        # Animations

        self.play(FadeIn(stereo_net))
        self.wait()

        self.wait()

        self.play(
            Create(orbits[0]),
            Create(orbits[1]),
            Create(orbits[2]),
            Create(orbits[3]),
            Create(orbits[4]),
            Create(orbits[5]),
            )
        self.wait()
        self.play(FadeOut(*orbits), FadeIn(vector_field))
        self.wait()

        self.play(
            ApplyPointwiseFunction(lambda x: np.exp(.4) * x, stereo_net),
            ApplyPointwiseFunction(lambda x: np.exp(.4) * x, vector_field),)
        self.wait()

        self.play(FadeOut(stereo_net, vector_field))

class XRotOrbits(ThreeDScene):
    def construct(self):

        # Camera

        phi = 75
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(1.5)

        # Mobjects

        punctured_sphere = Surface(lambda u,v: np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)]),
                         u_range = [0, 2 * np.pi],
                         v_range = [np.pi/12, np.pi],
                         resolution = (12, 11),
                         fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE, PURPLE)

        orbits = []
        thetas = np.arange(0, np.pi, np.pi/12)
            # Reuse thetas from stars
        for theta in thetas:
            if theta == np.pi/2:
                t_range = [np.pi/12, 2 * np.pi - np.pi/12]
            else:
                t_range = [0, 2 * np.pi]
            orbit = ParametricFunction(
                lambda t: np.array([np.cos(theta), np.sin(t) * np.sin(theta), np.cos(t) * np.sin(theta), ]),
                t_range = t_range,
            )
            orbits.append(orbit)

        vector_field = VGroup()
        for theta in thetas:
            if theta == np.pi/2:
                t_range = [np.pi/12, 2 * np.pi - np.pi/12]
                phis = np.arange(t_range[0], t_range[1], np.pi/6)
            else:
                t_range = [0, 2 * np.pi]
                phis = np.arange(t_range[0], t_range[1], np.pi/6)
            for phi in phis:
                x = np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi) * np.sin(theta)

                X = x / (1 - z)
                Y = y/ (1-z)
                Z = 0

                basepoint = np.array([X,Y,Z])
                vec_array = np.array([2 * X * Y, 1 - X**2 + Y**2, 0])/6
                vec_base = basepoint + 0.5 * vec_array
                vector = Vector(vec_array,
                                color = interpolate_color(BLUE_D, YELLOW, (x + 1)/2)
                                ).set_x(vec_base[0]).set_y(vec_base[1]).set_z(vec_base[2])
                vector_field.add(vector)

        # Transformations

        b_x = np.array([[0, 1],
                        [1, 0]])

        r_x = complex(0,1) * b_x

        R_x = expm(r_x/2 * np.pi/2)

        active_matrix = R_x

        # Animations

        self.play(FadeIn(punctured_sphere))

        self.play( 
            FadeIn(*orbits)
            )
        self.wait()

        self.play(
            ApplyPointwiseFunction(stereographic_projection, punctured_sphere),
            ApplyPointwiseFunction(stereographic_projection, orbits[0]),
            ApplyPointwiseFunction(stereographic_projection, orbits[1]),
            ApplyPointwiseFunction(stereographic_projection, orbits[2]),
            ApplyPointwiseFunction(stereographic_projection, orbits[3]),
            ApplyPointwiseFunction(stereographic_projection, orbits[4]),
            ApplyPointwiseFunction(stereographic_projection, orbits[5]),
            ApplyPointwiseFunction(stereographic_projection, orbits[6]),
            ApplyPointwiseFunction(stereographic_projection, orbits[7]),
            ApplyPointwiseFunction(stereographic_projection, orbits[8]),
            ApplyPointwiseFunction(stereographic_projection, orbits[9]),
            ApplyPointwiseFunction(stereographic_projection, orbits[10]),
            ApplyPointwiseFunction(stereographic_projection, orbits[11]),    
            )
        self.wait()

        self.begin_ambient_camera_rotation(rate = -75*DEGREES, about = 'phi')
        self.wait()
        self.stop_ambient_camera_rotation(about = 'phi')
        self.wait()

        self.play(FadeIn(vector_field))
        self.wait()

        self.play(FadeOut(punctured_sphere, vector_field, *orbits))

class XBoostOrbits(ThreeDScene):
    def construct(self):

        # Camera

        phi = 75
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(1.5)

        # Mobjects

        punctured_sphere = Surface(lambda u,v: np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)]),
                         u_range = [0, 2 * np.pi],
                         v_range = [np.pi/12, np.pi],
                         resolution = (12, 11),
                         fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE, PURPLE)

        orbits = []
        phis = np.arange(0, 2 * np.pi, np.pi/6)
        for phi in phis:
            if phi == 0:
                t_range_1 = [0, np.pi/2 - np.pi/12]
                orbit_1 = ParametricFunction(
                    lambda t: np.array([np.cos(t), np.sin(phi) * np.sin(t), np.cos(phi) * np.sin(t)]),
                    t_range = t_range_1,
                )
                orbits.append(orbit_1)
                t_range_2 = [np.pi/2 + np.pi/12, np.pi]
                orbit_2 = ParametricFunction(
                    lambda t: np.array([np.cos(t), np.sin(phi) * np.sin(t), np.cos(phi) * np.sin(t)]),
                    t_range = t_range_2,
                )
                orbits.append(orbit_2)
            else:
                t_range = [0, np.pi]
                orbit = ParametricFunction(
                    lambda t: np.array([np.cos(t), np.sin(phi) * np.sin(t), np.cos(phi) * np.sin(t)]),
                    t_range = t_range,
                )
                orbits.append(orbit)

        vector_field = VGroup()
        for phi in phis:
            if phi == 0 or phi == np.pi:
                t_range_1 = [0, np.pi/2 - np.pi/6]
                thetas_1 = np.arange(t_range_1[0], t_range_1[1], np.pi/12)

                t_range_2 = [np.pi/2 + np.pi/6, np.pi]
                thetas_2 = np.arange(t_range_2[0], t_range_2[1], np.pi/12)

                thetas = np.concatenate((thetas_1, thetas_2))
            else:
                t_range = [0, np.pi]
                thetas = np.arange(t_range[0], t_range[1], np.pi/12)
            for theta in thetas:
                x = np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi) * np.sin(theta)

                X = x/(1-z)
                Y = y/(1-z)
                Z = 0

                basepoint = np.array([X,Y,Z])
                vec_array = np.array([1 - X**2 + Y**2, - 2 * X * Y, 0])/6
                vec_base = basepoint #+ 0.5 * vec_array
                vector = Vector(vec_array,
                                color = interpolate_color(BLUE_D, YELLOW, (x + 1)/2)
                                ).set_x(vec_base[0]).set_y(vec_base[1]).set_z(vec_base[2])
                vector_field.add(vector)

        rot_orbits = []
        thetas = np.arange(0, np.pi, np.pi/12)
            # Reuse thetas from stars
        for theta in thetas:
            if theta == np.pi/2:
                t_range = [np.pi/12, 2 * np.pi - np.pi/12]
            else:
                t_range = [0, 2 * np.pi]
            rot_orbit = ParametricFunction(
                lambda t: np.array([np.cos(theta)/ (1 - np.cos(t) * np.sin(theta)), np.sin(t) * np.sin(theta) / (1 - np.cos(t) * np.sin(theta)), 0, ]),
                t_range = t_range,
            )
            rot_orbits.append(rot_orbit)

        # Animations

        self.play(FadeIn(punctured_sphere))

        self.play( 
            FadeIn(*orbits)
            )
        self.wait()

        self.play(
            ApplyPointwiseFunction(stereographic_projection, punctured_sphere),
            ApplyPointwiseFunction(stereographic_projection, orbits[0]),
            ApplyPointwiseFunction(stereographic_projection, orbits[1]),
            ApplyPointwiseFunction(stereographic_projection, orbits[2]),
            ApplyPointwiseFunction(stereographic_projection, orbits[3]),
            ApplyPointwiseFunction(stereographic_projection, orbits[4]),
            ApplyPointwiseFunction(stereographic_projection, orbits[5]),
            ApplyPointwiseFunction(stereographic_projection, orbits[6]),
            ApplyPointwiseFunction(stereographic_projection, orbits[7]),
            ApplyPointwiseFunction(stereographic_projection, orbits[8]),
            ApplyPointwiseFunction(stereographic_projection, orbits[9]),
            ApplyPointwiseFunction(stereographic_projection, orbits[10]),
            ApplyPointwiseFunction(stereographic_projection, orbits[11]),
            ApplyPointwiseFunction(stereographic_projection, orbits[12]),    
            )
        self.wait()

        self.begin_ambient_camera_rotation(rate = -75*DEGREES, about = 'phi')
        self.wait()
        self.stop_ambient_camera_rotation(about = 'phi')
        self.wait()

        self.play(FadeIn(*rot_orbits))
        self.wait()

        self.play(FadeOut(*rot_orbits))
        self.wait()

        self.play(FadeIn(vector_field))
        self.wait()

        self.play(FadeOut(punctured_sphere, *orbits, vector_field))

class XBoostPlusYRot(ThreeDScene):
    def construct(self):

        # Camera

        phi = 0
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(1.5)

        # Mobjects

        stereo_net = Surface(lambda u,v: np.array([np.cos(u) * (np.sin(v)/(1-np.cos(v))), np.sin(u) * (np.sin(v)/(1-np.cos(v))), 0]),
                         u_range = [0, 2 * np.pi],
                         v_range = [np.pi/12, np.pi],
                         resolution = (12, 11),
                         fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE, PURPLE)

        rot_orbits = []
        thetas = np.arange(0, np.pi, np.pi/12)
            # Reuse thetas from stars
        for theta in thetas:
            if theta == np.pi/2:
                t_range = [np.pi/12, 2 * np.pi - np.pi/12]
            else:
                t_range = [0, 2 * np.pi]
            rot_orbit = ParametricFunction(
                lambda t: np.array([np.sin(t) * np.sin(theta) / (1 - np.cos(t) * np.sin(theta)), np.cos(theta)/ (1 - np.cos(t) * np.sin(theta)), 0, ]),
                t_range = t_range,
                color = BLUE,
            )
            rot_orbits.append(rot_orbit)

        rot_vector_field = VGroup()
        for theta in thetas:
            if theta == np.pi/2:
                t_range = [np.pi/12, 2 * np.pi - np.pi/12]
                phis = np.arange(t_range[0], t_range[1], np.pi/6)
            else:
                t_range = [0, 2 * np.pi]
                phis = np.arange(t_range[0], t_range[1], np.pi/6)
            for phi in phis:
                x = np.sin(phi) * np.sin(theta)
                y = np.cos(theta)
                z = np.cos(phi) * np.sin(theta)

                X = x / (1 - z)
                Y = y/ (1-z)
                Z = 0

                basepoint = np.array([X,Y,Z])
                vec_array = np.array([1 + X**2 - Y**2, 2 * X * Y, 0])/6
                vec_base = basepoint #+ 0.5 * vec_array
                vector = Vector(vec_array,
                                color = BLUE
                                ).set_x(vec_base[0]).set_y(vec_base[1]).set_z(vec_base[2])
                rot_vector_field.add(vector)

        boost_orbits = []
        phis = np.arange(0, 2 * np.pi, np.pi/6)
        for phi in phis:
            if phi == 0:
                t_range_1 = [0, np.pi/2 - np.pi/12]
                orbit_1 = ParametricFunction(
                    lambda t: np.array([np.cos(t)/ (1 - np.cos(phi) * np.sin(t)), np.sin(phi) * np.sin(t)/ (1 - np.cos(phi) * np.sin(t)), 0]),
                    t_range = t_range_1,
                    color = RED
                )
                boost_orbits.append(orbit_1)
                t_range_2 = [np.pi/2 + np.pi/12, np.pi]
                orbit_2 = ParametricFunction(
                    lambda t: np.array([np.cos(t)/ (1 - np.cos(phi) * np.sin(t)), np.sin(phi) * np.sin(t)/ (1 - np.cos(phi) * np.sin(t)), 0]),
                    t_range = t_range_2,
                    color = RED
                )
                boost_orbits.append(orbit_2)
            else:
                t_range = [0, np.pi]
                orbit = ParametricFunction(
                    lambda t: np.array([np.cos(t)/ (1 - np.cos(phi) * np.sin(t)), np.sin(phi) * np.sin(t)/ (1 - np.cos(phi) * np.sin(t)), 0]),
                    t_range = t_range,
                    color = RED
                )
                boost_orbits.append(orbit)

        boost_vector_field = VGroup()
        for phi in phis:
            if phi == 0 or phi == np.pi:
                t_range_1 = [0, np.pi/2 - np.pi/6]
                thetas_1 = np.arange(t_range_1[0], t_range_1[1], np.pi/12)

                t_range_2 = [np.pi/2 + np.pi/6, np.pi]
                thetas_2 = np.arange(t_range_2[0], t_range_2[1], np.pi/12)

                thetas = np.concatenate((thetas_1, thetas_2))
            else:
                t_range = [0, np.pi]
                thetas = np.arange(t_range[0], t_range[1], np.pi/12)
            for theta in thetas:
                x = np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi) * np.sin(theta)

                X = x/(1-z)
                Y = y/(1-z)
                Z = 0

                basepoint = np.array([X,Y,Z])
                vec_array = np.array([1 - X**2 + Y**2, - 2 * X * Y, 0])/6
                vec_base = basepoint #+ 0.5 * vec_array
                vector = Vector(vec_array,
                                color = RED
                                ).set_x(vec_base[0]).set_y(vec_base[1]).set_z(vec_base[2])
                boost_vector_field.add(vector)

        translation_orbits = []
        ys = np.arange(-4, +4, 1)
        for y in ys:
            x_bound = np.floor(np.sqrt((1 + np.cos(np.pi/12))/(1 - np.cos(np.pi/12)) - y**2))
            orbit = ParametricFunction(
                lambda t: np.array([t, y, 0]),
                t_range = [-x_bound, +x_bound],
                color = GREEN
            )
            translation_orbits.append(orbit)

        translation_vector_field = VGroup()
        xs = np.arange(-8, +8, 1)
        for y in ys:
            for x in xs:
                if x**2 + y**2 > (1 + np.cos(np.pi/12))/(1 - np.cos(np.pi/12)):
                    continue
                basepoint = np.array([x, y, 0])
                vec_array = np.array([2, 0, 0])/6
                vec_base = basepoint
                vector = Vector(
                    vec_array,
                    color = GREEN
                ).set_x(vec_base[0]).set_y(vec_base[1]).set_z(vec_base[2])
                translation_vector_field.add(vector)

        # Animations

        self.play(FadeIn(boost_vector_field))
        self.wait()

        self.play(FadeIn(rot_vector_field))
        self.wait()

        self.play(FadeOut(boost_vector_field, rot_vector_field), FadeIn(translation_vector_field))
        self.wait()

        self.play(FadeIn(stereo_net))
        self.wait()

        self.begin_ambient_camera_rotation(rate = 75 * DEGREES, about = 'phi')
        self.wait()
        self.stop_ambient_camera_rotation(about = 'phi')

        self.play(
            ApplyPointwiseFunction(stereographic_inverse, stereo_net),
            ApplyPointwiseFunction(stereographic_inverse, translation_vector_field),
            ApplyPointwiseFunction(stereographic_inverse, translation_orbits[0]),
            ApplyPointwiseFunction(stereographic_inverse, translation_orbits[1]),
            ApplyPointwiseFunction(stereographic_inverse, translation_orbits[2]),
            ApplyPointwiseFunction(stereographic_inverse, translation_orbits[3]),
            ApplyPointwiseFunction(stereographic_inverse, translation_orbits[4]),
            ApplyPointwiseFunction(stereographic_inverse, translation_orbits[5]),
            ApplyPointwiseFunction(stereographic_inverse, translation_orbits[6]),
            ApplyPointwiseFunction(stereographic_inverse, translation_orbits[7]),
            )
        self.wait()

        self.play(
            FadeOut(stereo_net, translation_vector_field, *translation_orbits)
        )

class AddVecFields(ThreeDScene):
    def construct(self):

        # Camera

        phi = 0
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(1.5)

        # Mobjects

        stereo_net = Surface(lambda u,v: np.array([np.cos(u) * (np.sin(v)/(1-np.cos(v))), np.sin(u) * (np.sin(v)/(1-np.cos(v))), 0]),
                         u_range = [0, 2 * np.pi],
                         v_range = [np.pi/12, np.pi],
                         resolution = (12, 11),
                         fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE, PURPLE)

        r_translation_orbits = []
        ys = np.arange(-4, +4, 1)
        for y in ys:
            x_bound = np.floor(np.sqrt((1 + np.cos(np.pi/12))/(1 - np.cos(np.pi/12)) - y**2))
            orbit = ParametricFunction(
                lambda t: np.array([t, y, 0]),
                t_range = [-x_bound, +x_bound],
                color = RED
            )
            r_translation_orbits.append(orbit)

        r_translation_vector_field = VGroup()
        xs = np.arange(-8, +8, 1)
        for y in ys:
            for x in xs:
                if x**2 + y**2 > (1 + np.cos(np.pi/12))/(1 - np.cos(np.pi/12)):
                    continue
                basepoint = np.array([x, y, 0])
                vec_array = np.array([2, 0, 0])/6
                vec_base = basepoint+ 1/2 * vec_array
                vector = Vector(
                    vec_array,
                    color = RED
                ).set_x(vec_base[0]).set_y(vec_base[1]).set_z(vec_base[2])
                r_translation_vector_field.add(vector)

        u_translation_orbits = []
        ys = np.arange(-4, +4, 1)
        for y in ys:
            x_bound = np.floor(np.sqrt((1 + np.cos(np.pi/12))/(1 - np.cos(np.pi/12)) - y**2))
            orbit = ParametricFunction(
                lambda t: np.array([t, y, 0]),
                t_range = [-x_bound, +x_bound],
                color = BLUE
            )
            u_translation_orbits.append(orbit)

        u_translation_vector_field = VGroup()
        xs = np.arange(-8, +8, 1)
        for y in ys:
            for x in xs:
                if x**2 + y**2 > (1 + np.cos(np.pi/12))/(1 - np.cos(np.pi/12)):
                    continue
                basepoint = np.array([x, y, 0])
                vec_array = np.array([0, 2, 0])/6
                vec_base = basepoint+ 1/2 * vec_array
                vector = Vector(
                    vec_array,
                    color = BLUE
                ).set_x(vec_base[0]).set_y(vec_base[1]).set_z(vec_base[2])
                u_translation_vector_field.add(vector)

        d_translation_orbits = []
        ys = np.arange(-4, +4, 1)
        for y in ys:
            x_bound = np.floor(np.sqrt((1 + np.cos(np.pi/12))/(1 - np.cos(np.pi/12)) - y**2))
            orbit = ParametricFunction(
                lambda t: np.array([t, y, 0]),
                t_range = [-x_bound, +x_bound],
                color = GREEN
            )
            d_translation_orbits.append(orbit)

        d_translation_vector_field = VGroup()
        xs = np.arange(-8, +8, 1)
        for y in ys:
            for x in xs:
                if x**2 + y**2 > (1 + np.cos(np.pi/12))/(1 - np.cos(np.pi/12)):
                    continue
                basepoint = np.array([x, y, 0])
                vec_array = np.array([2, 2, 0])/6
                vec_base = basepoint + 1/2 * vec_array
                vector = Vector(
                    vec_array,
                    color = GREEN
                ).set_x(vec_base[0]).set_y(vec_base[1]).set_z(vec_base[2])
                d_translation_vector_field.add(vector)

        # Animations

        self.play(FadeIn(r_translation_vector_field))
        self.wait()

        self.play(FadeIn(u_translation_vector_field))
        self.wait()

        self.play(FadeIn(d_translation_vector_field))
        self.wait()

        self.play(FadeOut(r_translation_vector_field, u_translation_vector_field, d_translation_vector_field))





class SanityCheck(ThreeDScene):
    def construct(self):

        # Camera

        phi = 75
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(2)

        # Mobjects

        sphere = Surface(lambda u,v: np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)]),
                         u_range = [0, 2 * np.pi],
                         v_range = [0, np.pi],
                         resolution = (12, 12),
                         fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE, PURPLE)

        # Animations

        self.play(FadeIn(sphere))
        self.wait()

        self.play(Rotate(sphere, angle = np.pi/12, axis = np.array([0,1,0])))
        self.wait()

        xi = -np.pi/12 * 1/2
        self.play(
            ApplyPointwiseFunction(lambda x: conformal_transformation_3_1(x, Lambda = SL_to_SO(np.array([[np.cosh(xi),np.sinh(xi)],[np.sinh(xi), np.cosh(xi)]]))), sphere)
        )
        self.wait()

        self.play(FadeOut(sphere))
        self.wait()

class ConformalBoundary(ThreeDScene):
    def construct(self):
        
        # Camera
        phi = 75
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(1.5)

        # Mobjects
        upper_hyperboloid = Surface(
            lambda u,v: np.array([np.sinh(v) * np.cos(u), np.sinh(v) * np.sin(u), np.cosh(v)]),
            resolution = (16,16),
            u_range = [0, 2 * np.pi],
            v_range = [-3,3],
            fill_opacity = 0.5
        ).set_fill_by_checkerboard(BLUE,BLUE)

        celestial_circle = ParametricFunction(
            lambda t: np.array([np.cos(t), np.sin(t), 1]),
            t_range = [0, 2 * np.pi],
            color = YELLOW
        )

        # Animations

        self.play(FadeIn(upper_hyperboloid, celestial_circle))
        self.wait()

        self.play(ApplyPointwiseFunction(
            lambda x: 10 * x,
            celestial_circle
        ))
        self.wait()

        def south_pole_stereog(x):
            x, y, z = x
            X = x/(1+z)
            Y = y/(1+z)
            return np.array([X, Y, 0])

        self.play(
            ApplyPointwiseFunction(
                south_pole_stereog,
                celestial_circle
            ),
            ApplyPointwiseFunction(
                south_pole_stereog,
                upper_hyperboloid
            )
        )
        self.wait()

        self.play(FadeOut(upper_hyperboloid, celestial_circle))

class SpinorPath(ThreeDScene):
    def construct(self):
        
        # Camera
        phi = 75
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(1.5)

        # Mobjects
        sphere = Surface(lambda u,v: np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)]),
                         u_range = [0, 2 * np.pi],
                         v_range = [0, np.pi],
                         resolution = (12, 12),
                         fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE, PURPLE)

        path_1 = ParametricFunction(
            lambda t: np.array([0, 0, t]),
            t_range = [0, 1],
            color = PURPLE_A
        )

        path_2 = ParametricFunction(
            lambda t: np.array([0, 0, t]),
            t_range = [-1, 0],
            color = PURPLE_A
        )

        # Animations
        self.play(FadeIn(sphere))
        self.wait()

        self.play(Create(path_1))
        self.play(Create(path_2))
        self.wait()

        self.play(FadeOut(sphere, path_1, path_2))

class HeartMobiusMap(ThreeDScene):
    def construct(self):
        
        # Camera
        phi = 0
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(1)

        # Mobjects
        heart = ParametricFunction(
            lambda t: np.array([16 * (np.sin(t) **3), 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t), 0]),
            t_range = [0, 2 * np.pi],
            color = PINK
        )

        hearts = VGroup()

        for size in np.arange(0.01, 0.1, 0.01):
            heart = ParametricFunction(
            lambda t: size * np.array([16 * (np.sin(t) **3), 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t), 0]),
            t_range = [0, 2 * np.pi],
            color = interpolate_color(PURE_RED, PINK, size * 10)
            )
            hearts.add(heart)

        

        # Animations
        self.play(FadeIn(hearts))
        self.wait()

        self.play(ApplyComplexFunction(lambda z: ( z + np.sqrt(3) * complex(0,1))/(z*complex(0,1)*np.sqrt(3) + 1), hearts))
        self.wait()

        self.play(ApplyComplexFunction(lambda z: ( z + np.sqrt(3) * complex(0,1))/(z*complex(0,1)*np.sqrt(3) + 1), hearts))
        self.wait()

        self.play(ApplyComplexFunction(lambda z: ( z + np.sqrt(3) * complex(0,1))/(z*complex(0,1)*np.sqrt(3) + 1), hearts))
        self.wait()

        self.play(FadeOut(hearts))

class StereoNetInversion(ThreeDScene):
    def construct(self):

        # Camera

        phi = 0
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(1.5)

        # Mobjects

        stereo_net = Surface(lambda u, v: np.array([np.cos(u) * np.sin(v)/(1 - np.cos(v)), np.sin(u) * np.sin(v)/(1 - np.cos(v)), 0]),
                             u_range = [0, 2 * np.pi],
                             v_range = [np.pi/24, np.pi  - np.pi/24],
                             resolution = (12, 11),
                             fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE, PURPLE)

        stars = Mobject1D()
        n_phis = 6
        n_thetas = 12
        phis = np.linspace(0, 2 * np.pi, n_phis, endpoint=False)
        thetas = np.linspace(np.pi/24, np.pi - np.pi/24, n_thetas, endpoint=True)
        for phi in phis:
            for theta in thetas:
                x = np.cos(phi) * np.sin(theta)/(1 - np.cos(theta))
                y = np.sin(phi) * np.sin(theta)/(1 - np.cos(theta))
                z = 0
                stars.add_points([np.array([x, y, z])])

        map_text = Tex(r'$w \mapsto \frac{1}{w}$').set_y(-2)

        # Animations
        self.play(FadeIn(stereo_net))
        self.add(stars)
        self.wait()

        self.play(Write(map_text))
        self.wait()

        self.play(ApplyPointwiseFunction(lambda x: x/np.linalg.norm(x)**2, stereo_net),
                  ApplyPointwiseFunction(lambda x: x/np.linalg.norm(x)**2, stars))
        self.wait()

        self.play(ApplyPointwiseFunction(lambda x: np.array([x[0], -x[1], 0]), stereo_net),
                  ApplyPointwiseFunction(lambda x: np.array([x[0], -x[1], 0]), stars))
        self.wait()

        self.remove(stars)
        self.play(Unwrite(map_text), FadeOut(stereo_net))

class CelestialSphereInversion(ThreeDScene):
    def construct(self):

        # Camera

        phi = 75
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(1.5)

        # Mobjects

        punctured_sphere = Surface(lambda u, v: np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)]),
                             u_range = [0, 2 * np.pi],
                             v_range = [np.pi/24, np.pi  - np.pi/24],
                             resolution = (12, 11),
                             fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE, PURPLE)

        stars = Mobject1D()
        n_phis = 6
        n_thetas = 12
        phis = np.linspace(0, 2 * np.pi, n_phis, endpoint=False)
        thetas = np.linspace(np.pi/24, np.pi - np.pi/24, n_thetas, endpoint=True)
        for phi in phis:
            for theta in thetas:
                x = np.cos(phi) * np.sin(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(theta)
                stars.add_points([np.array([x, y, z])])

        # Animations
        self.play(FadeIn(punctured_sphere))
        self.add(stars)
        self.wait()

        self.play(ApplyPointwiseFunction(lambda x: np.array([x[0], x[1], -x[2]]), punctured_sphere),
                  ApplyPointwiseFunction(lambda x: np.array([x[0], x[1], -x[2]]), stars))
        self.wait()

        self.play(ApplyPointwiseFunction(lambda x: np.array([x[0], -x[1], x[2]]), punctured_sphere),
                  ApplyPointwiseFunction(lambda x: np.array([x[0], -x[1], x[2]]), stars))
        self.wait()

        self.remove(stars)
        self.play(FadeOut(punctured_sphere))

class StereoNetTranslation(ThreeDScene):
    def construct(self):

        # Camera

        phi = 0
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(1.5)

        # Mobjects

        stereo_net = Surface(lambda u, v: np.array([np.cos(u) * np.sin(v)/(1 - np.cos(v)), np.sin(u) * np.sin(v)/(1 - np.cos(v)), 0]),
                             u_range = [0, 2 * np.pi],
                             v_range = [np.pi/24, np.pi  - np.pi/24],
                             resolution = (12, 11),
                             fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE, PURPLE)

        stars = Mobject1D()
        n_phis = 6
        n_thetas = 12
        phis = np.linspace(0, 2 * np.pi, n_phis, endpoint=False)
        thetas = np.linspace(np.pi/24, np.pi - np.pi/24, n_thetas, endpoint=True)
        for phi in phis:
            for theta in thetas:
                x = np.cos(phi) * np.sin(theta)/(1 - np.cos(theta))
                y = np.sin(phi) * np.sin(theta)/(1 - np.cos(theta))
                z = 0
                stars.add_points([np.array([x, y, z])])

        map_text = Tex(r'$w \mapsto w + b$').set_y(-2)

        # Animations
        self.play(FadeIn(stereo_net))
        self.add(stars)
        self.wait()

        self.play(Write(map_text))
        self.wait()

        self.play(ApplyPointwiseFunction(lambda x: x + np.array([1,0,0]), stereo_net),
                  ApplyPointwiseFunction(lambda x: x + np.array([1,0,0]), stars))
        self.wait()

        self.remove(stars)
        self.play(Unwrite(map_text), FadeOut(stereo_net))