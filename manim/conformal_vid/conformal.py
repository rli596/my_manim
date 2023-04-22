'''
conformal.py
'''

# Dependencies
from manim import *
import numpy as np

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
        t_f = Tex(r'Isometries').set_x(2).set_y(.5)

        t_c_white = Tex(r'Dilatation').set_x(-3).set_y(-.5).set_color(WHITE)
        t_a_grey = Tex(r'Translation').set_x(-3).set_y(1.5).set_color(GREY_E)
        t_g = Tex(r'Linear conformal transformations').set_x(2).set_y(.5)

        t_a_white = Tex(r'Translation').set_x(-3).set_y(1.5).set_color(WHITE)
        t_h = Tex(r'Special conformal transformations').set_x(-3).set_y(-1.5)
        t_i = Tex(r'"Conformal symmetries\\ of 2d space"').set_x(2).set_y(.5)

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
        self.play(ApplyComplexFunction(lambda z:0, text),
                  ApplyComplexFunction(lambda z:0, t_a), 
                  ApplyComplexFunction(lambda z:0, t_b), 
                  ApplyComplexFunction(lambda z:0, t_c),
                  ApplyComplexFunction(lambda z:0, t_h), 
                  ApplyComplexFunction(lambda z:0, t_i))
        self.wait()