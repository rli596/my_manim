###
#
# sg.py
#
# Scenes for sine-gordon
#
###

'''
Dependencies
'''
from manim import *
from sg_utils import *

'''
Scenes
'''

class Tractroid(ThreeDScene):
    def construct(self):
        # Camera
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(0.5)

        # Mobjects
        tractroid_mobject = Surface(tractroid, resolution = (16,16), v_range=[0,2 * np.pi], u_range = [-5,5])

        self.add(tractroid_mobject)

class DiniSurface(ThreeDScene):
    def construct(self):
        # Camera
        phi = 75
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(1)

        # Mobjects
        dini_1 = Surface(lambda u,v: dini_surface(u, v, -1, 0.15), resolution = (16,16), u_range = [0,2*np.pi], v_range=[np.pi/24,np.pi - np.pi/24])

        # Animations
        self.add(dini_1,
        )

class DiniSurfaceHomotopy(ThreeDScene):
    def construct(self):
        # Camera
        phi = 75
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(1)

        # Mobjects
        epsilon = 1e-3
        tractroid_mobject = Surface(tractroid, resolution = (16,16), v_range=[-np.pi + epsilon, np.pi - epsilon], u_range = [-5,5])

        # Animations
        self.play(FadeIn(tractroid_mobject))
        self.wait()
        self.play(Homotopy(pseudosphere_homotopy, tractroid_mobject))
        self.wait()

class DiniSurfaceIncision(ThreeDScene):
    def construct(self):
        # Camera
        phi = 75
        theta = 120
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(1)

        # Mobjects
        epsilon = 1e-3
        tractroid_mobject = Surface(tractroid, resolution = (16,16), v_range=[-np.pi + epsilon, np.pi - epsilon], u_range = [-5,5], fill_opacity = 0.7)

        incision = ParametricFunction(lambda t: tractroid(t, np.pi), t_range = [-5, 5]).set_color(RED)

        # Animations
        self.play(FadeIn(tractroid_mobject))
        self.wait()
        self.play(Create(incision))
        self.wait()
        self.play(FadeOut(incision))
        self.wait()
        self.play(Homotopy(pseudosphere_homotopy, tractroid_mobject))
        self.wait()

class DiniSurfaceIncisionWikipedia(ThreeDScene):
    def construct(self):
        # Camera
        phi = 75
        theta = 150
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(2)

        # Mobjects
        epsilon = 1e-3
        tractroid_bottom = Surface(tractroid, resolution = (8,16), v_range=[-np.pi + epsilon, np.pi - epsilon], u_range = [-5,0], fill_opacity = 0.5).set_fill_by_checkerboard(PURPLE, PURPLE)
        tractroid_top = Surface(tractroid, resolution = (8,16), v_range=[-np.pi + epsilon, np.pi - epsilon], u_range = [0,5], fill_opacity = 0.5).set_fill_by_checkerboard(PURPLE, PURPLE)

        incision = ParametricFunction(lambda t: tractroid(-t, np.pi), t_range = [-4, 4]).set_color(WHITE)

        # Animations
        self.add(tractroid_bottom, tractroid_top)
        self.wait()
        self.play(Create(incision), run_time = 4)
        self.wait()
        self.play(FadeOut(incision))
        self.wait()
        self.play(Homotopy(lambda x, y, z, t: pseudosphere_homotopy(x,y,z,2*t), tractroid_bottom),
                  Homotopy(lambda x, y, z, t: pseudosphere_homotopy(x,y,z,2*t), tractroid_top))
        self.wait()
        self.play(FadeOut(tractroid_top))
        self.wait()
        self.play(FadeIn(tractroid_top))
        self.wait()
        self.play(Homotopy(lambda x, y, z, t: pseudosphere_homotopy(x,y,z,-2*t), tractroid_bottom),
                  Homotopy(lambda x, y, z, t: pseudosphere_homotopy(x,y,z,-2*t), tractroid_top))

class Breather(ThreeDScene):
    def construct(self):
        # Camera
        phi = 75
        theta = 30
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        # Mobjects
        breather = Surface(lambda u,v: breather_surface(u,v,3/5), 
                           resolution = (16, 30), 
                           u_range=[-15,15], 
                           v_range = [0,2 * np.pi * 5],
                           fill_opacity = 0.7)

        self.add(breather)

class OneSoliton3D(ThreeDScene):
    def construct(self):
        # Camera
        phi = 0
        theta = 75
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        # Mobjects
        rapidity = 0.5
        soliton = ParametricFunction(lambda t: np.array([t, np.cos(one_soliton(np.cosh(rapidity) * t)), np.sin(one_soliton(np.cosh(rapidity) * t))]),
                                     t_range = [-10,10],
                                     )

        self.play(Create(soliton))
        self.wait()
        self.play(Homotopy(lambda x,y,z,t: np.array([x + np.sinh(rapidity) * t, y, z]), soliton), rate_functions = linear)
        self.wait()

class OneSolitonRibbonSurface(ThreeDScene):
    def construct(self):
        # Camera
        phi = 75
        theta = 120
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        # Mobjects
        L = 10

        ribbon = VGroup()
        ribbon_surface = Surface(lambda u,v: np.array([u, v, 0]),
                                 u_range = [-L, L],
                                 v_range = [-2, 2],
                                 fill_opacity= 0.5,
                                 resolution = (4 * L,4)).set_fill_by_checkerboard(GREEN, GREEN)
        ribbon.add(ribbon_surface)
        ribbon_line = ParametricFunction(lambda t: np.array([t, 0, 2]),
                                         t_range = [-L, L],
                                         ).set_color(RED)
        ribbon.add(ribbon_line)
        for x_value in range(-L + 1, L):
            vector = Vector(np.array([0,0,2])).set_x(x_value).set_color(BLUE)
            ribbon.add(vector)


        self.add(ribbon)
        self.play(Homotopy(soliton_homotopy, ribbon, rate_functions = linear, run_time = 5))

class BreatherRibbonSurface(ThreeDScene):
    def construct(self):
        # Camera
        phi = 75
        theta = 120
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        # Mobjects
        L = 10

        ribbon = VGroup()
        ribbon_surface = Surface(lambda u,v: np.array([u, v, 0]),
                                 u_range = [-L, L],
                                 v_range = [-2, 2],
                                 fill_opacity= 0.7,
                                 resolution = (8 * L,2),
                                 checkerboard_colors = [GREEN, GREEN])
        ribbon.add(ribbon_surface)
        ribbon_line = ParametricFunction(lambda t: np.array([t, 0, 2]),
                                         t_range = [-L, L],
                                         ).set_color(RED)
        ribbon.add(ribbon_line)
        for x_value in range(-L + 1, L):
            vector = Vector(np.array([0,0,2])).set_x(x_value).set_color(BLUE)
            ribbon.add(vector)


        self.add(ribbon)
        self.play(Homotopy(lambda x,y,z,t: soliton_homotopy(x,y,z,t,breather, t_range = [-4 * np.pi, 4 * np.pi]), ribbon, rate_functions = linear).set_run_time(10))
        self.wait()

class TwoSolitonRibbonSurface(ThreeDScene):
    def construct(self):
        # Camera
        phi = 75
        theta = 120
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        # Mobjects
        L = 10

        ribbon = VGroup()
        ribbon_surface = Surface(lambda u,v: np.array([u, v, 0]),
                                 u_range = [-L, L],
                                 v_range = [-2, 2],
                                 fill_opacity= 0.7,
                                 resolution = (8 * L,2),
                                 checkerboard_colors = [GREEN, GREEN])
        ribbon.add(ribbon_surface)
        ribbon_line = ParametricFunction(lambda t: np.array([t, 0, 2]),
                                         t_range = [-L, L],
                                         ).set_color(RED)
        ribbon.add(ribbon_line)
        for x_value in range(-L + 1, L):
            vector = Vector(np.array([0,0,2])).set_x(x_value).set_color(BLUE)
            ribbon.add(vector)


        self.add(ribbon)
        self.play(Homotopy(lambda x,y,z,t: soliton_homotopy(x,y,z,t,two_soliton, t_range = [-20,20]), ribbon, rate_functions = linear).set_run_time(20))
        self.wait()

class KinkAntikinkRibbonSurface(ThreeDScene):
    def construct(self):
        # Camera
        phi = 75
        theta = 120
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        # Mobjects
        L = 10

        ribbon = VGroup()
        ribbon_surface = Surface(lambda u,v: np.array([u, v, 0]),
                                 u_range = [-L, L],
                                 v_range = [-2, 2],
                                 fill_opacity= 0.7,
                                 resolution = (4 * L,2),
                                 checkerboard_colors = [GREEN, GREEN])
        ribbon.add(ribbon_surface)
        ribbon_line = ParametricFunction(lambda t: np.array([t, 0, 2]),
                                         t_range = [-L, L],
                                         ).set_color(RED)
        ribbon.add(ribbon_line)
        for x_value in range(-L + 1, L):
            vector = Vector(np.array([0,0,2])).set_x(x_value).set_color(BLUE)
            ribbon.add(vector)


        self.add(ribbon)
        self.play(Homotopy(lambda x,y,z,t: soliton_homotopy(x,y,z,t,kink_antikink, t_range = [-20,20]), ribbon, rate_functions = linear).set_run_time(20))
        self.wait()

# Scenes

class OneSoliton(ThreeDScene):
    def construct(self):
        # Camera
        phi = 0
        theta = 90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(0.5)

        # Mobjects
        soliton = ParametricFunction(lambda t: np.array([t, 0, 0]),
                                     t_range = [-10, 10])
        envelope = VGroup()
        upper_env = ParametricFunction(lambda t: np.array([t, 2 * np.pi, 0]),
                                     t_range = [-10, 10]).set_color(RED)
        lower_env = ParametricFunction(lambda t: np.array([t, - 2 * np.pi, 0]),
                                     t_range = [-10, 10]).set_color(RED)
        mid_env = ParametricFunction(lambda t: np.array([t, 0, 0]),
                                     t_range = [-10, 10]).set_color(GREEN)
        envelope.add(upper_env, lower_env, mid_env)


        self.add(envelope)
        self.add(soliton)
        self.play(Homotopy(lambda x,y,z,t: soliton_plane_homotopy(x,y,z,t, func = boosted_one_soliton), soliton, rate_functions = linear).set_run_time(10))

class KdV(Scene):
    def construct(self):
        # Parameters
        time_range = [-3, 3]
        # Mobjects
        f = ParametricFunction(lambda t: np.array([t,-3,0]),
                               t_range = [-7, 7])
        # Functions
        KdV_solution = lambda x,t: 6 * (3 + 4 * np.cosh(2 * x - 8 * t) + np.cosh(4 * x - 64 * t)) / (3 * np.cosh(x - 28 * t) + np.cosh(3 * x - 36 * t))**2
            # Two-soliton formula from https://young.physics.ucsc.edu/250/mathematica/soliton.nb.pdf
        KdV_homotopy = lambda x,y,z,t: np.array([x, KdV_solution(x,t * (time_range[1] - time_range[0]) + time_range[0]) - 3, z])
        # Animations
        self.play(Homotopy(KdV_homotopy, f).set_run_time(40))