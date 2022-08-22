###
# mobius_text.py
#
# text scenes for mobius maps part of video
#
###

'''
Dependencies
'''

from manim import *

'''
Scenes
'''

class SingleText(Scene):
    def construct(self):
        scale = 2
        dyn_text = Tex(r'Spacetime symmetries\\$\cong$\\Möbius maps\\$\cong$\\$\text{SL}(2, \mathbb{C})\mathbb{Z_2}').scale(scale)

        self.play(Write(dyn_text))
        self.wait()
        self.play(Unwrite(dyn_text))
        self.wait()

class MobiusMapsText(ThreeDScene):
    def construct(self):

        phi = 0
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(1.5)

        dyn_text = Tex(r'Möbius maps: $f(w) = \frac{aw + b}{cw + d}$').set_y(1)
        dyn_text_2 = MathTex(r'a, b, c, d \in \mathbb{C}').set_y(-1)

        mapsto_form_text = MathTex(r'w \mapsto \frac{aw + b}{cw + d}').set_y(1)
        function_form_text = MathTex(r'f(w) = \frac{aw + b}{cw + d}').set_y(1)

        mapsto_eg_rot_text = Tex(r'Rotation: $w \mapsto e^{i\theta}w$').set_y(-1)
        mapsto_eg_dil_text = Tex(r'Dilatation: $w \mapsto \alpha w$').set_y(-1)
        mapsto_eg_inv_text = Tex(r'Inversion: $w \mapsto \frac{1}{w}$').set_y(-1)
        mapsto_eg_transl_text = Tex(r'Translation: $w \mapsto w + b$').set_y(-1)

        mapsto_infinity_text = MathTex(r'f\left(-\frac{d}{c}\right) := \infty ?').set_y(-1)
        mapsfrom_infinity_c_zero_text = Tex(r'$f(\infty) := \infty$ if $c = 0$').set_y(-1)
        mapsfrom_infinity_c_nonzero_text = Tex(r'$f(\infty) := \frac{a}{c}$ if $c \neq 0$').set_y(-1)

        self.play(Write(dyn_text))
        self.wait()
        
        self.play(Write(dyn_text_2))
        self.wait()

        self.play(Transform(dyn_text, mapsto_form_text))
        self.wait()

        self.play(Transform(dyn_text_2, mapsto_eg_rot_text))
        self.wait()

        self.play(Transform(dyn_text_2, mapsto_eg_dil_text))
        self.wait()

        self.play(Transform(dyn_text_2, mapsto_eg_inv_text))
        self.wait()

        self.play(Transform(dyn_text_2, mapsto_eg_transl_text))
        self.wait()

        self.play(Transform(dyn_text, function_form_text))
        self.wait()

        self.play(Transform(dyn_text_2, mapsto_infinity_text))
        self.wait()

        self.play(Transform(dyn_text_2, mapsfrom_infinity_c_zero_text))
        self.wait()

        self.play(Transform(dyn_text_2, mapsfrom_infinity_c_nonzero_text))
        self.wait()

        self.play(FadeOut(dyn_text, dyn_text_2))

class ChangingTheta(Scene):
    def construct(self):
        scale = 2
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

class IsomorphismsText(Scene):
    def construct(self):
        scale = 2
        dyn_text = Tex(r'Spacetime symmetries').scale(scale).set_y(2)
        equiv_text_1 = Tex(r'$\cong$').scale(scale).set_y(1)
        dyn_text_2 = Tex(r'Möbius maps').scale(scale).set_y(0)
        equiv_text_2 = Tex(r'$\cong$').scale(scale).set_y(-1)
        dyn_text_3 = Tex(r'$\text{SL}(2, \mathbb{C})\/\mathbb{Z}_2$').scale(scale).set_y(-2)

        spacetime_group_text = Tex(r'$\text{SO}(1,3)^{\uparrow}$').set_y(2).scale(scale)
        mobius_group_text = Tex(r'$\text{PSL}(2, \mathbb{C})$').set_y(0).scale(scale)

        self.play(Write(dyn_text))
        self.play(Write(equiv_text_1))
        self.play(Write(dyn_text_2))
        self.wait()

        self.play(Transform(dyn_text, spacetime_group_text))
        self.play(Transform(dyn_text_2, mobius_group_text))
        self.wait()

        self.play(Write(equiv_text_2))
        self.play(Write(dyn_text_3))
        self.wait()

        self.play(
            Unwrite(dyn_text), 
            Unwrite(dyn_text_2),
            Unwrite(dyn_text_3),
            Unwrite(equiv_text_1),
            Unwrite(equiv_text_2),)