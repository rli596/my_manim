'''
conformal.py
'''

# Dependencies
from manim import *
import numpy as np
from conformal_utils import *

# Animation classes
class Example(Scene):
    def construct(self):
        scale = 1
        dyn_text = Tex(r'$\theta = 0$').scale(scale)

        rot_text_1 = Tex(r'$\theta = \frac{2}{3}\pi$').scale(scale)
        rot_text_2 = Tex(r'$\theta = \frac{4}{3}\pi$').scale(scale)
        rot_text_3 = Tex(r'$\theta = 2\pi$').scale(scale)

        self.play(Write(dyn_text))
        self.wait()
        self.play(Transform(dyn_text, rot_text_1))
        self.wait()
        self.play(Transform(dyn_text, rot_text_2))
        self.wait()
        self.play(Transform(dyn_text, rot_text_3))
        self.wait()
        self.play(Unwrite(dyn_text))
        self.wait()

class Intro(Scene):
    def construct(self):
        scale = 1
        title = Tex(r'Conformal symmetry').scale(100)

        a = Tex(r'Conformal field theory (CFT)').scale(scale).set_y(2.5)
        b = Tex(r'AdS-CFT correspondence').set_y(0.5)
        c = Tex(r'String theory is an example of CFT').set_y(-1.5)
        d = Tex(r'"A theory that has conformal symmetry"').set_y(0)

        e = Tex(r'Math: angle-preserving transformation').set_y(2.5)
        f = Tex(r'Physics: symmetries of a scale-invariant system').set_y(0)

        self.play(ApplyComplexFunction(lambda z: z/50, title))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: 0, title))
        self.wait()
        self.play(Write(a))
        self.wait()
        self.play(Write(b))
        self.wait()
        self.play(Write(c))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: 0, b), ApplyComplexFunction(lambda z: 0, c))
        self.wait()
        self.play(Write(d))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: 0, a), ApplyComplexFunction(lambda z: 0, d))
        self.wait()
        self.play(Write(e))
        self.wait()
        self.play(Write(f))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: 0, e), ApplyComplexFunction(lambda z: 0, f))
        self.wait()

class Intro2(Scene):
    def construct(self):
        woman= ImageMobject("woman-standing.png").scale(0.01)
        a = Tex(r'Stress $\sigma$').set_x(-4).set_y(1).scale(2)
        b = Tex(r'$\frac{\text{Force}}{\text{Area}}$').set_x(4).set_y(1).scale(2)

        c = Tex(r'$\sim \frac{\text{Volume}}{\text{Area}}$').set_x(4).set_y(1).scale(2)
        d = Tex(r'$\sim \frac{\text{Length}^3}{\text{Length}^2}$').set_x(4).set_y(1).scale(2)
        e = Tex(r'$\sim \text{Length}$').set_x(4).set_y(1).scale(2)

        self.play(ApplyComplexFunction(lambda z: 50 * z,woman))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: 2 * z + 1 * complex(0,1),woman))
        self.wait()
        self.play(Write(a))
        self.wait()
        self.play(Write(b))
        self.wait()
        self.play(Transform(b,c))
        self.wait()
        self.play(Transform(b,d))
        self.wait()
        self.play(Transform(b,e))
        self.wait()
        self.play(ApplyComplexFunction(lambda z:0, a), ApplyComplexFunction(lambda z:0, b), ApplyComplexFunction(lambda z:0, woman))
        self.wait()

class Conf1(Scene):
    def construct(self):
        text = Tex(r'Conformal transformations on the plane').set_y(2.5).set_color(YELLOW)

        line1 = Line(start=np.array([0,0,0]), end=2*np.array([1,0,0])/100).set_color(BLUE)
        line2 = Line(start=np.array([0,0,0]), end=2*np.array([4,3,0])/500).set_color(BLUE)
        angle = Angle(line1, line2, radius=0.006).set_color(YELLOW)
        a = Group(line1, line2, angle)
        
        t_a = Tex(r'Translation').set_x(-2)
        t_b = Tex(r'Rotation').set_x(-2)
        t_c = Tex(r'Dilatation').set_x(-2)

        #Animations
        self.play(Write(text))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: 100*z, a))
        self.wait()
        self.play(Write(t_a))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: z + complex(2,1), a))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: z - complex(2,1), a))
        self.wait()
        self.play(Transform(t_a, t_b))
        self.wait()
        self.play(Homotopy(lambda x, y, z, t: np.array([np.cos(t) * x - np.sin(t) * y, np.sin(t) * x + np.cos(t) * y, 0]), a))
        self.wait()
        self.play(Homotopy(lambda x, y, z, t: np.array([np.cos(-t) * x - np.sin(-t) * y, np.sin(-t) * x + np.cos(-t) * y, 0]), a))
        self.wait()
        self.play(Transform(t_a, t_c))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: 3/2*z, a))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: 2/3*z, a))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: 3/2*(z+complex(1,-1)), a))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: 2/3*z - complex(1,-1), a))
        self.wait()
        self.play(ApplyComplexFunction(lambda z:0, a), Unwrite(text), Unwrite(t_a))
        self.wait()

class Skew(Scene):
    def construct(self):
        t_a = Tex(r'Skew/shear transformation').set_y(2.5)
        t_b = Tex(r'Funky transformation').set_y(2.5)
        
        number_plane = NumberPlane(
            x_range=(-3, 3, 1),
            y_range=(-3, 3, 1),
            x_length=2/100,
            y_length=2/100,
        ).set_y(0)

        line1 = Line(start=np.array([0,0,0]), end=2*np.array([1,0,0]))
        line2 = Line(start=np.array([0,0,0]), end=2*np.array([0,1,0]))
        line3 = Line(start=np.array([0,0,0]), end=2*np.array([1,1,0]))
        angle = Angle(line1, line2)
        angle2 = Angle(line1, line3)

        self.wait()
        self.play(Write(t_a))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: 180*z, number_plane))
        self.wait()
        self.play(Create(angle))
        self.wait()
        self.play(FadeOut(angle))
        self.wait()
        self.play(Homotopy(lambda x,y,z,t: np.array([x + t * y, y, z]), number_plane))
        self.wait()
        self.play(Create(angle2))
        self.wait()
        self.play(FadeOut(angle2))
        self.wait()
        self.play(Homotopy(lambda x,y,z,t: np.array([x - t * y, y, z]), number_plane))
        self.wait()
        self.play(Transform(t_a, t_b))
        self.wait()
        self.play(Homotopy(lambda x,y,z,t: np.array([x + t/4 * np.sin(20 * y), y + t/4 * np.sin(20 * x), z]), number_plane))
        self.wait()
        self.play(ApplyComplexFunction(lambda z:0, number_plane), ApplyComplexFunction(lambda z: complex(0, 2.5), t_a))
        self.wait()

class Conf2(Scene):
    def construct(self):
        text = Tex(r'Conformal transformations on the plane').set_y(2.5).set_color(YELLOW)
        
        t_a = Tex(r'Translation').set_x(-3).set_y(1.5)
        t_b = Tex(r'Rotation').set_x(-3).set_y(.5)
        t_c = Tex(r'Dilatation').set_x(-3).set_y(-.5)

        t_d = Tex(r'This is \textbf{all} \\conformal transformations!').set_x(2).set_y(.5)

        t_e = Tex(r'Reflection?').set_x(-3).set_y(-1.5)

        line1 = Line(start=np.array([0,0,0]), end=2*np.array([1,0,0])/100).set_color(BLUE)
        line2 = Line(start=np.array([0,0,0]), end=2*np.array([4,3,0])/500).set_color(BLUE)
        angle = Angle(line1, line2, radius=0.006).set_color(YELLOW)
        a = Group(line1, line2, angle)

        t_c_grey = Tex(r'Dilatation').set_x(-3).set_y(-.5).set_color(GREY_E)
        t_f = Tex(r'Isometries').set_x(2).set_y(.5).set_color(RED)

        t_c_white = Tex(r'Dilatation').set_x(-3).set_y(-.5).set_color(WHITE)
        t_a_grey = Tex(r'Translation').set_x(-3).set_y(1.5).set_color(GREY_E)
        t_g = Tex(r'Linear maps').set_x(2).set_y(.5).set_color(RED)

        t_a_white = Tex(r'Translation').set_x(-3).set_y(1.5).set_color(WHITE)
        t_h = Tex(r'Special conformal transformations').set_x(-3).set_y(-1.5)
        t_i = Tex(r'"Conformal symmetries\\ of 2d space"').set_x(2).set_y(.5).set_color(RED)
        t_j = Tex(r'Symmetries of 3+1 spacetime').set_x(2).set_y(-.5).set_color(RED_A)
        t_k = Tex(r'Conformal symmetries\\of extended 2d space').set_x(2).set_y(.5).set_color(RED)

        #Animations
        self.wait()
        self.play(Write(text))
        self.wait()
        self.play(Write(t_a), Write(t_b), Write(t_c))
        self.wait()
        self.play(Write(t_d))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: complex(2,.5), t_d))
        self.wait()
        self.play(Write(t_e))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: 100*z, a))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: np.conjugate(z), a))
        self.wait()
        self.play(ApplyComplexFunction(lambda z:complex(-3,-1.5), t_e), ApplyComplexFunction(lambda z:0, a))
        self.wait()
        self.play(Transform(t_c, t_c_grey))
        self.wait()
        self.play(Write(t_f))
        self.wait()
        self.play(Transform(t_c,t_c_white), Unwrite(t_f), Transform(t_a, t_a_grey))
        self.wait()
        self.play(Write(t_g))
        self.wait()
        self.play(Transform(t_a,t_a_white), Unwrite(t_g), Write(t_h))
        self.wait()
        self.play(Write(t_i))
        self.wait()
        self.play(Write(t_j))
        self.wait()
        self.play(Transform(t_i, t_k))
        self.wait()
        self.play(ApplyComplexFunction(lambda z:0, text),
                  ApplyComplexFunction(lambda z:0, t_a), 
                  ApplyComplexFunction(lambda z:0, t_b), 
                  ApplyComplexFunction(lambda z:0, t_c),
                  ApplyComplexFunction(lambda z:0, t_h), 
                  ApplyComplexFunction(lambda z:0, t_i),
                  ApplyComplexFunction(lambda z:0, t_j))
        self.wait()

class Conf3(ThreeDScene):
    def construct(self):
        # Camera
        phi = 0
        theta = -90 * DEGREES
        self.set_camera_orientation(phi = phi, theta = theta)

        # Mobjects
        disc = Surface(lambda u,v: np.array([u**2 * np.sin(v), u**2 * np.cos(v), 0]),
                       u_range = [0, 3],
                       v_range = [0, 2 * np.pi],
                       fill_color=BLUE_D,
                       resolution = (12,12),
                       fill_opacity=0.7)
        pole = Dot3D(np.array([0,0,2]),
                     color=YELLOW)
        t = Tex(r'Conformal compactification').set_y(-2.5)
        
        # Animations
        self.wait()
        self.play(FadeIn(disc))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: 0.5 * z, disc))
        self.wait()
        self.begin_ambient_camera_rotation(rate = np.pi/6, about = "phi")
        self.wait(2)
        self.stop_ambient_camera_rotation(about = 'phi')
        self.wait()
        self.play(ApplyPointwiseFunction(isp, disc))
        self.wait()
        xi = 2
        self.play(Homotopy(lambda x,y,z,t: 2 * np.array([x / (2 * np.cosh(xi * t) + z * np.sinh(xi * t)), 
                                                     y / (2 * np.cosh(xi * t) + z * np.sinh(xi * t)), 
                                                     (z * np.cosh(xi * t) + 2 * np.sinh(xi * t)) / (2 * np.cosh(xi * t) + z * np.sinh(xi * t))]), disc))
        self.wait()
        self.play(FadeIn(pole))
        self.wait()
        self.begin_ambient_camera_rotation(rate = -np.pi/6, about = "phi")
        self.wait(2)
        self.stop_ambient_camera_rotation(about = 'phi')
        self.wait()
        self.play(Write(t))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: 0, disc),
                  ApplyComplexFunction(lambda z: 0, pole),
                  ApplyComplexFunction(lambda z: complex(0,-2.5), t),)
        self.wait()

class Conf4(Scene):
    def construct(self):
        circle = ParametricFunction(lambda t: 2* np.array([np.cos(t), np.sin(t), 0]),
                                    t_range = [0, 2 * np.pi],
                                    color = GREEN)
        t_circ = Tex('Unit circle').set_color(GREEN).set_x(0).set_y(2.5)
        real_line = ParametricFunction(lambda t: np.array([t, 0, 0]),
                                       t_range = [-7, 7])
        south_pole = Dot(point = 2* np.array([0, -1, 0]),
                         color = RED)
        north_pole = Dot(point = 2 * np.array([0, +1, 0]),
                         color = RED)
        t_inf = Tex('Point at infinity').set_color(RED).set_y(2.5)

        arc = ParametricFunction(lambda t: 2* np.array([np.cos(t), np.sin(t), 0]),
                                t_range = [- np.pi/2, np.pi/2],
                                color = RED)
        
        arc_img = ParametricFunction(lambda t: np.array([t, 0, 0]),
                                       t_range = [0, 100],
                                       color = RED)
        
        t_length = Tex('Length = $L$').set_y(-.5)
        t_length_2 = Tex('Length? = $2L$?').set_y(-.5)
        
        self.wait()
        self.play(Create(circle), Write(t_circ))
        self.wait()
        self.play(Unwrite(t_circ), Create(real_line))
        self.wait()
        self.play(FadeIn(south_pole))
        self.wait()
        self.play(FadeIn(north_pole), Write(t_inf))
        self.wait()
        self.play(Unwrite(t_inf))
        self.wait()
        self.play(Create(arc), Unwrite(t_inf))
        self.wait()
        self.play(Transform(arc, arc_img), FadeOut(south_pole, north_pole))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: 0, circle),
                  ApplyComplexFunction(lambda z: 0, arc),
                  ApplyComplexFunction(lambda z: z / 7, real_line))
        self.wait()
        self.play(Write(t_length))
        self.wait()
        self.play(Transform(t_length, t_length_2), ApplyComplexFunction(lambda z: 2 * z, real_line))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: 0, t_length),
                  ApplyComplexFunction(lambda z: 0, real_line))
        self.wait()

class Conf5(ThreeDScene):
    def construct(self):
        phi = 0
        theta = -90 * DEGREES
        self.set_camera_orientation(phi=phi, theta = theta)

        # Mobjects
        up_arrow = Vector(direction = -UP)
        right_arrow = Vector(direction = RIGHT)
        disc = Surface(lambda u,v: u * np.array([np.cos(v), np.sin(v), 0]),
                       u_range = [0, 5],
                       v_range = [0, 2 * np.pi],
                       fill_opacity= 0.5,
                       resolution = (16,12))
        angle = Angle(up_arrow, right_arrow)

        t_1 = Tex('1. How do we describe compactification precisely?').set_y(2.3)
        t_2 = Tex('2. What are angles on a sphere?').set_y(-2.3)
        # Animations
        self.wait()
        self.play(Create(up_arrow), Create(right_arrow), Create(angle))
        self.wait()
        self.begin_ambient_camera_rotation(rate = np.pi/3, about = "phi")
        self.wait()
        self.stop_ambient_camera_rotation(about = "phi")
        self.wait()
        self.play(Create(disc))
        self.wait()
        self.play(ApplyPointwiseFunction(isp, disc),
                  ApplyPointwiseFunction(isp, up_arrow),
                  ApplyPointwiseFunction(isp, right_arrow),
                  ApplyPointwiseFunction(isp, angle))
        self.wait()
        self.begin_ambient_camera_rotation(rate = -np.pi/3, about = "phi")
        self.wait()
        self.stop_ambient_camera_rotation(about = "phi")
        self.wait()
        self.play(Write(t_1))
        self.wait()
        self.play(Write(t_2))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: 0, disc),
                  ApplyComplexFunction(lambda z: 0, up_arrow),
                  ApplyComplexFunction(lambda z: 0, right_arrow),
                  ApplyComplexFunction(lambda z: 0, angle),
                  ApplyComplexFunction(lambda z: 0, t_1),
                  ApplyComplexFunction(lambda z: 0, t_2),
                  )
        self.wait()

class Conf6(ThreeDScene):
    def construct(self):
        phi = 0
        theta = -90 * DEGREES
        self.set_camera_orientation(phi=phi, theta = theta)

        # Mobjects
        t = Tex(r'2: Define angles between \textit{curves}').set_y(3)
        curve_1 = ParametricFunction(lambda t: np.array([4, 3,0]) + 5 * np.array([np.cos(t), np.sin(t), 0]),
                                     t_range = [np.pi + np.pi/12, np.pi + np.pi/3])
        curve_2 = ParametricFunction(lambda t: np.array([-3, 4,0]) + 5 * np.array([np.cos(t), np.sin(t), 0]),
                                     t_range = [-np.pi/2, - np.pi/6])
        sphere = Surface(lambda u,v: 2 * np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)]),
                         u_range = [0, 2 * np.pi],
                         v_range = [0, np.pi],
                         fill_opacity= 0.5,
                         resolution = (24, 24))
        
        # Animations
        self.wait()
        self.play(Write(t))
        self.wait()
        self.play(Create(curve_1), Create(curve_2))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: z - complex(0,1),curve_1),
                  ApplyComplexFunction(lambda z: z - complex(0,1),curve_2))
        self.wait()
        self.play(FadeOut(t))
        self.begin_ambient_camera_rotation(np.pi/3, about='phi')
        self.wait()
        self.stop_ambient_camera_rotation(about='phi')
        self.wait()
        self.play(ApplyPointwiseFunction(isp, curve_1),
                  ApplyPointwiseFunction(isp, curve_2))
        self.wait()
        self.play(FadeIn(sphere))
        self.wait()
        self.play(FadeOut(sphere))
        self.wait()
        self.play(FadeOut(curve_1), FadeOut(curve_2))
        self.wait()

class Conf7(Scene):
    def construct(self):

        # Mobjects
        t = Tex(r'2: Define angles between \textit{curves}').set_y(3)
        curve_1 = ParametricFunction(lambda t: np.array([4, 3,0]) + 5 * np.array([np.cos(t), np.sin(t), 0]),
                                     t_range = [np.pi + np.pi/12, np.pi + np.pi/3],
                                     color = BLUE)
        curve_2 = ParametricFunction(lambda t: np.array([-3, 4,0]) + 5 * np.array([np.cos(t), np.sin(t), 0]),
                                     t_range = [-np.pi/2, - np.pi/6],
                                     color = RED)
        v_1 = Vector(np.array([-3, 4,0])/3, color = BLUE)
        v_2 = Vector(np.array([4,3,0])/3, color = RED)
        angle = Angle(v_2, v_1)

        t = Tex('Works for 3D curves as well!').set_y(3)     
        # Animations
        self.wait()
        self.play(Create(curve_1), Create(curve_2))
        self.wait()
        self.play(Create(v_1), Create(v_2))
        self.wait()
        #self.play(FadeOut(curve_1), FadeOut(curve_2))
        #self.wait()
        self.play(Create(angle))
        self.wait()
        self.play(Write(t))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: 0, t),
                  ApplyComplexFunction(lambda z: 0, v_1),
                  ApplyComplexFunction(lambda z: 0, v_2),
                  ApplyComplexFunction(lambda z: 0, angle),
                  ApplyComplexFunction(lambda z: 0, curve_1),
                  ApplyComplexFunction(lambda z: 0, curve_2))
        self.wait()

class Conf8(Scene):
    def construct(self):
        t_a = Tex(r'Want to define $\pi: S^2 \dashrightarrow \mathbb{R}^2$').set_y(2)
        t_b = Tex(r'$S^2 = $ sphere, $\mathbb{R}^2 = $ plane, $\dashrightarrow = $ partial mapping').set_y(1)
        t_c = Tex(r'\textit{Stereographic projection}', color = YELLOW)

        circle = ParametricFunction(lambda t: np.array([np.cos(t), np.sin(t), 0]),
                                    t_range = [0, 2 * np.pi],
                                    color = GREEN)
        line = ParametricFunction(lambda t: np.array([t, 0, 0]),
                                  t_range = [-7, 7],)
        circle_stat = ParametricFunction(lambda t: np.array([np.cos(t), np.sin(t), 0]),
                                    t_range = [- 3 * np.pi/2, np.pi/2],
                                    color = RED)
        circle_dyn = ParametricFunction(lambda t: np.array([np.cos(t), np.sin(t), 0]),
                                    t_range = [- 3 * np.pi/2, np.pi/2],
                                    color = RED)
        line_proj = ParametricFunction(lambda t: np.array([t, 0, 0]),
                                  t_range = [-20, 20],
                                  color = RED)
        north_pole = Dot(np.array([0,1,0]),
                         color = RED)
        arb_pt = Dot(np.array([np.sqrt(3)/2, 1/2, 0]),
                     color = RED)
        line_ray = ParametricFunction(lambda t: (1 - t) * np.array([0, 1, 0]) + t * np.array([np.sqrt(3)/2, 1/2, 0]),
                                      t_range = [-7, 14],)
        img_pt = Dot(np.array([np.sqrt(3), 0, 0]),
                     color = RED)
        t_d = Tex(r'$(x,y) \mapsto \frac{x}{1 - y}$').set_y(-1.5)


        # Animations
        self.wait()
        self.play(Write(t_a), Write(t_b))
        self.wait()
        self.play(Write(t_c))
        self.wait()
        self.play(ApplyComplexFunction(lambda z:0, t_a),
                  ApplyComplexFunction(lambda z:0, t_b),
                  ApplyComplexFunction(lambda z:0, t_c),)
        self.wait()
        self.play(Create(circle), Create(line))
        self.wait()
        self.play(Transform(circle_dyn, line_proj))
        self.wait()
        self.play(Transform(circle_dyn, circle_stat))
        self.wait()
        self.play(FadeIn(north_pole), FadeIn(arb_pt))
        self.wait()
        self.play(Create(line_ray))
        self.wait()
        self.play(FadeIn(img_pt))
        self.wait()
        self.play(Write(t_d))
        self.wait()
        self.play(ApplyComplexFunction(lambda z:0, t_d),
                  ApplyComplexFunction(lambda z:0, circle),
                  ApplyComplexFunction(lambda z:0, line),
                  ApplyComplexFunction(lambda z:0, circle_dyn),
                  ApplyComplexFunction(lambda z:0, north_pole),
                  ApplyComplexFunction(lambda z:0, arb_pt),
                  ApplyComplexFunction(lambda z:0, line_ray),
                  ApplyComplexFunction(lambda z:0, img_pt),)
        self.wait()

class Conf9(ThreeDScene):
    def construct(self):
        # Camera
        phi = 60 * DEGREES
        theta = 0 * DEGREES
        self.set_camera_orientation(phi = phi, theta = theta)

        # Mobjects
        disc = Surface(lambda u,v: 2 * (10 - u) * np.array([np.cos(v), np.sin(v), 0]),
                       u_range = [0, 10],
                       v_range = [0, 2 * np.pi],
                       resolution = (24, 24), 
                       fill_opacity= 0.5)
        sphere = Surface(lambda u,v: 2 * np.array([np.cos(v) * np.sin(u), np.sin(v) * np.sin(u), np.cos(u)]),
                         u_range = [0, np.pi],
                         v_range = [0, 2 * np.pi],
                         resolution=(24,24), 
                         fill_opacity=.5)
        
        curve1 = ParametricFunction(lambda t: 2 * np.array([np.cos(-t) * np.sin(-t + np.pi/2), np.sin(-t) * np.sin(-t + np.pi/2), np.cos(-t + np.pi/2)]),
                                    t_range = [-.5,.5],
                                    color = RED)
        curve2 = ParametricFunction(lambda t: 2 * np.array([np.cos(t) * np.sin(-t + np.pi/2), np.sin(t) * np.sin(-t + np.pi/2), np.cos(-t + np.pi/2)]),
                                    t_range = [-.5,.5],
                                    color = BLUE)
        
        vec1 = Arrow(start = np.array([2,0,0]), end = np.array([2,-1,1]), color = RED)
        vec2 = Arrow(start = np.array([2,0,0]), end = np.array([2,1,1]), color = BLUE)
        angle = Angle(vec1, vec2)

        new_vec1 = Arrow(start = np.array([2,0,0]), end = np.array([1, 1, 0]), color = RED)
        new_vec2 = Arrow(start = np.array([2,0,0]), end = np.array([1, -1, 0]), color = BLUE)
        new_angle = Angle(new_vec1, new_vec2)

        # Animations
        self.wait()
        self.play(FadeIn(sphere))
        self.wait()
        self.play(Create(curve1), Create(curve2))
        self.wait()
        self.play(Create(vec1), Create(vec2))
        self.wait()
        self.play(Uncreate(vec1), Uncreate(vec2))
        self.wait()
        self.play(FadeOut(sphere))
        self.wait()
        self.play(ApplyPointwiseFunction(stereographic_projection, curve1),
                  ApplyPointwiseFunction(stereographic_projection, curve2),)
        self.wait()
        self.play(FadeIn(disc))
        self.wait()
        self.begin_ambient_camera_rotation(rate = -np.pi/6, about = 'phi')
        self.wait(2)
        self.stop_ambient_camera_rotation(about = 'phi')
        self.wait()
        self.play(Create(new_vec1), Create(new_vec2), Create(new_angle))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: 0, disc),
                  ApplyComplexFunction(lambda z: 0, new_vec1),
                  ApplyComplexFunction(lambda z: 0, new_vec2),
                  ApplyComplexFunction(lambda z: 0, new_angle),
                  ApplyComplexFunction(lambda z: 0, curve1),
                  ApplyComplexFunction(lambda z: 0, curve2))
        self.wait()

class Conf10(Scene):
    def construct(self):
        t1 = Tex(r'2D plane conformally extends to sphere').set_y(+2)
        t2 = Tex(r'Conformal transformations of plane extend to\\ transformations of sphere')

        self.wait()
        self.play(Write(t1))
        self.wait()
        self.play(Write(t2))
        self.wait()

class Conf11(ThreeDScene):
    def construct(self):
        phi = -120 * DEGREES
        theta = -90 * DEGREES
        self.set_camera_orientation(phi = phi, theta = theta)

        disc = Surface(lambda u, v: np.tan(u/2) * np.array([np.cos(v), np.sin(v), 0]),
                       u_range = [0, np.pi - np.pi/6],
                       v_range = [0, 2 * np.pi], 
                       fill_opacity= 0.7,
                       resolution = (24,24))
        
        self.wait()
        self.play(FadeIn(disc))
        self.wait()
        self.play(Homotopy(lambda x,y,z,t: np.array([np.cos(t * np.pi/3) * x  - np.sin(t * np.pi/3) * y, np.cos(t * np.pi/3) * y + np.sin(t * np.pi/3) * x, z]),
                           disc))
        self.wait()
        self.play(ApplyPointwiseFunction(isp, disc))
        self.wait()
        self.play(Homotopy(lambda x,y,z,t: np.array([np.cos(t * np.pi/3) * x  + np.sin(t * np.pi/3) * y, np.cos(t * np.pi/3) * y - np.sin(t * np.pi/3) * x, z]),
                           disc))
        self.wait()
        self.play(ApplyPointwiseFunction(stereographic_projection, disc))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: 1.2 * z, disc))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: 5/6 * z, disc))
        self.wait()
        self.play(ApplyPointwiseFunction(isp, disc))
        self.wait()
        xi = 0.6
        self.play(Homotopy(lambda x,y,z,t: 2 * np.array([x / (2 * np.cosh(xi * t) + z * np.sinh(xi * t)), 
                                                     y / (2 * np.cosh(xi * t) + z * np.sinh(xi * t)), 
                                                     (z * np.cosh(xi * t) + 2 * np.sinh(xi * t)) / (2 * np.cosh(xi * t) + z * np.sinh(xi * t))]), disc))
        self.wait()
        xi = -0.6
        self.play(Homotopy(lambda x,y,z,t: 2 * np.array([x / (2 * np.cosh(xi * t) + z * np.sinh(xi * t)), 
                                                     y / (2 * np.cosh(xi * t) + z * np.sinh(xi * t)), 
                                                     (z * np.cosh(xi * t) + 2 * np.sinh(xi * t)) / (2 * np.cosh(xi * t) + z * np.sinh(xi * t))]), disc))
        self.wait()
        self.play(ApplyPointwiseFunction(stereographic_projection, disc))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: z + 1, disc))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: z - 1, disc))
        self.wait()
        self.play(ApplyPointwiseFunction(isp, disc))
        self.wait()
        self.play(Homotopy(lambda x,y,z,t: isp(stereographic_projection([x,y,z]) + 3* np.array([t,0,0])), disc))
        self.wait()
        self.play(FadeOut(disc))
        self.wait()

class Conf12(Scene):
    def construct(self):
        t1 = Tex(r'Conformal symmetries of the plane').set_y(2).set_color(YELLOW)
        t2 = Tex(r'=').set_y(1).set_color(YELLOW)
        t3 = Tex(r'Rotation, dilatation and translation').set_y(0).set_color(YELLOW)

        t4 = Tex(r'Plane $\rightarrow$ Sphere (Conformal extension)').set_y(-1).set_color(BLUE)

        t5 = Tex(r'Conformal symmetries of extended plane preserving point at $\infty$').set_y(2).set_color(YELLOW)

        t6 = Tex(r'Conformal symmetries of extended plane').set_y(2).set_color(YELLOW)
        t7 = Tex(r'Rotation, dilatation, translation and \textbf{SCTs}').set_y(0).set_color(YELLOW)

        t8 = Tex(r'More to come in part 2').set_y(1).scale(100)
        t9 = Tex(r'Thanks for watching!').set_y(0).scale(100)

        self.wait()
        self.play(Write(t1), Write(t2), Write(t3))
        self.wait()
        self.play(Write(t4))
        self.wait()
        self.play(Transform(t1, t5))
        self.wait()
        self.play(Transform(t1, t6), Transform(t3, t7))
        self.wait()
        self.play(ApplyComplexFunction(lambda z: 0, t1),
                  ApplyComplexFunction(lambda z: 0, t2),
                  ApplyComplexFunction(lambda z: 0, t3),
                  ApplyComplexFunction(lambda z: 0, t4),)
        self.wait()
        self.play(ApplyComplexFunction(lambda z: z/100 + complex(0,1), t8),
                  ApplyComplexFunction(lambda z: z/100, t9))
        self.wait()

class Thumbnail(Scene):
    def construct(self):
        t1 = Tex(r'Invitation').set_x(-3).set_y(3).set_color(YELLOW).scale(3)
        t2 = Tex(r'to').set_x(-2).set_y(2).set_color(GREEN).scale(2)
        t3 = Tex(r'Conformal').set_x(-3).set_y(1).set_color(WHITE).scale(3)
        t4 = Tex(r'Field').set_x(-3).set_y(-0.5).set_color(WHITE).scale(4)
        t5 = Tex(r'Theory').set_x(-3).set_y(-2).scale(3)
        t6 = Tex(r'pt. 1').set_x(3).set_y(-2.5).scale(2)

        disc = Surface(lambda u, v: np.tan(u/2) * np.array([np.cos(v), np.sin(v), 0]) + np.array([5, 2, 0]),
                       u_range = [0, np.pi - np.pi/6],
                       v_range = [0, 2 * np.pi], 
                       fill_opacity= 0.7,
                       resolution = (24,24))

        self.add(t1, t2, t3, t4, t5, disc, t6)