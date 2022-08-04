###
# oneplusone.py
#
# Contains scenes for 1+1 dimensions part of video
###

'''
Dependencies
'''

from manim import *
from numpy import number
from lorentz_utils import *

'''
Scenes
'''

class ImpossibleTriangle(Scene):
    def construct(self):
        # Constants
        TRIANGLE_COLOR = RED

        # Camera

        # Mobjects
        '''
        These have been embedded into animations
        '''

        # Animations

        ## First showing the impossible triangle, applying Pythagoras theorem
        triangle = Polygon([-2,-2,0], 
                            [+2,-2,0],
                            [+2,+2,0],).set_color(TRIANGLE_COLOR)
        self.play(Create(triangle))

        one = MathTex("1").set_x(0).set_y(-2.5)
        self.play(Write(one))

        eye = MathTex("i").set_x(+2.5).set_y(+0)
        self.play(Write(eye))

        pythag = MathTex('1^2 + i^2 = 0^2').set_x(-2).set_y(+2)
        self.play(Write(pythag))

        zero = MathTex("0").set_x(-0.5).set_y(+0.5)
        self.play(Write(zero))

        self.play(Unwrite(one), Unwrite(eye), Unwrite(zero), Unwrite(pythag))

        ## Extending the impossible triangle to a pseudometric on complex space
        triangle_2 = Polygon([0,0,0], 
                            [1,0,0],
                            [1,1,0],).set_color(TRIANGLE_COLOR)
        self.play(Transform(triangle, triangle_2))

        number_plane = NumberPlane(
            x_range=[-5, +5, 1],
            y_range=[-3, +3, 1]).set_z_index(-1)
        self.play(Create(number_plane))
        
        ##
        # 
        one_2 = MathTex("1").set_x(1).set_y(-0.3)
        eye_2 = MathTex("i").set_x(-0.2).set_y(1)
        self.play(Write(one_2), Write(eye_2))
        pythag_2 = MathTex('1^2 + i^2 = 0').set_x(-3).set_y(-1.5)
        self.play(Write(pythag_2))

        # contribution from real part 
        pythag_x = MathTex('1^2 + 0^2 = 1').set_x(-3).set_y(-1.5)
        self.play(Transform(pythag_2, pythag_x))

        # contribution from imaginary part
        pythag_t = MathTex('0^2 + i^2 = -1').set_x(-3).set_y(-1.5)
        self.play(Transform(pythag_2, pythag_t))

        # another example: 2 + 3i
        triangle_3 = Polygon([0,0,0], 
                            [2,0,0],
                            [2,3,0],).set_color(TRIANGLE_COLOR)
        self.play(Transform(triangle, triangle_3))
        pythag_3 = MathTex('2^2 + (3i)^2 = -5').set_x(-3).set_y(-1.5)
        self.play(Transform(pythag_2, pythag_3))

        # general x + ti
        x_axis = MathTex("x").set_x(5.2).set_y(0.2)
        t_axis = MathTex("t").set_x(0.2).set_y(3.2)
        pythag_coord = MathTex('x^2 + (it)^2 = -t^2 + x^2').set_x(-3).set_y(-1.5)
        self.play(Write(x_axis), Write(t_axis), Transform(pythag_2, pythag_coord))
        self.wait(1)

        # getting rid of complex dependence
        self.play(Unwrite(one_2), Unwrite(eye_2))
        pythag_simp = MathTex('-t^2 + x^2').set_x(-3).set_y(-1.5)
        self.play(Transform(pythag_2, pythag_simp))

        self.wait()
        self.play(Unwrite(pythag_2), FadeOut(number_plane), FadeOut(x_axis), FadeOut(t_axis), FadeOut(triangle))
        self.wait()

class CompareEuclidean(Scene):
    def construct(self):

        # Create vector
        number_plane = NumberPlane(
            x_range=[-7, +7, 1],
            y_range=[-4, +4, 1]).set_z_index(-1)
        self.play(FadeIn(number_plane))

        vector = Vector([3,2])
        self.play(Create(vector))

        # Modulus
        dyn_text = MathTex("|x + iy|^2 = (x + iy)(x - iy) = x^2 + y^2", color = 'YELLOW').set_y(-3.5)
        modulus = MathTex("|x + iy|^2 = (x + iy)(x - iy) = x^2 + y^2").set_y(-3.5)
        self.play(Write(dyn_text))

        # Euclidean distance
        euclidean_text = MathTex(r"\text{Euclidean distance: }x^2 + y^2", color = 'YELLOW').set_y(-3.5)
        self.play(Transform(dyn_text, euclidean_text))

        # 
        positive = MathTex(r"x^2 + y^2 > 0 \text{ for } (x,y) \neq (0,0)", color = 'YELLOW').set_y(-3.5)
        self.play(Transform(dyn_text, positive))

        # zero norm
        zero_norm_vec = Vector([2,2])
        zero_norm = MathTex(r"-2^2 + 2^2 = 0", color = 'YELLOW').set_y(-3.5)
        self.play(Transform(vector, zero_norm_vec), Transform(dyn_text, zero_norm))

        # negative norm
        neg_norm_vec = Vector([2,3])
        zero_norm = MathTex(r"-3^2 + 2^2 < 0", color = 'YELLOW').set_y(-3.5)
        self.play(Transform(vector, neg_norm_vec), Transform(dyn_text, zero_norm))

        # pseudo distance
        pseudo_text = MathTex(r"\text{Pseudo-distance: }-t^2 + x^2", color = 'YELLOW').set_y(-3.5)
        self.play(Transform(dyn_text, pseudo_text))

        # Points of modulus 1
        x_vec = Vector([0,1])
        self.play(Transform(vector, x_vec))
        circle = Circle(radius = 1)
        circle_eqn = MathTex("x^2 + y^2 = 1", color = 'YELLOW').set_y(-3.5)
        self.play(Create(circle), FadeOut(vector), Transform(dyn_text, circle_eqn))

        # Points of pseudo-distance 1
        hyperbola_eqn = MathTex("-t^2 + x^2 = -1", color = 'YELLOW').set_y(-3.5)
        upper_hyperbola = ParametricFunction(lambda xi: [np.sinh(xi), np.cosh(xi), 0], t_range = [-10, 10], color = PURPLE)
        lower_hyperbola = ParametricFunction(lambda xi: [np.sinh(xi), -np.cosh(xi), 0], t_range = [-10, 10], color = PURPLE)
        self.play(Transform(dyn_text, hyperbola_eqn), Create(upper_hyperbola), Create(lower_hyperbola))

        # All circles are equivalent
        general_circle_eqn = MathTex("x^2 + y^2 = R^2", color = 'YELLOW').set_y(-3.5)
        self.play(Transform(dyn_text, general_circle_eqn), FadeOut(upper_hyperbola), FadeOut(lower_hyperbola))
        self.play(ScaleInPlace(circle, 3))
        self.play(ScaleInPlace(circle, 0.01))

        # Causal structure: timelike
        timelike_hyperbola_eqn = MathTex("-t^2 + x^2 = -T^2", color = 'YELLOW').set_y(-3.5)
        self.play(Transform(dyn_text, timelike_hyperbola_eqn), FadeOut(circle), FadeIn(upper_hyperbola), FadeIn(lower_hyperbola))
        hyperbolas = VGroup()
        hyperbolas.add(upper_hyperbola)
        hyperbolas.add(lower_hyperbola)
        self.play(ScaleInPlace(hyperbolas, 3))
        self.play(ScaleInPlace(hyperbolas, 1/3 * 0.2))
        self.play(ScaleInPlace(hyperbolas, 5))

        # spacelike
        spacelike_hyperbola_eqn = MathTex("-t^2 + x^2 = +X^2", color = 'YELLOW').set_y(-3.5)
        switch_matrix = [[0,1],[1,0]]
        self.play(ApplyMatrix(switch_matrix, hyperbolas), Transform(dyn_text, spacelike_hyperbola_eqn))

        # null
        null_eqn = MathTex("-t^2 + x^2 = 0", color = 'YELLOW').set_y(-3.5)
        self.play(Transform(dyn_text, null_eqn), ScaleInPlace(hyperbolas, 0.02))

        # fadeout
        self.play(FadeOut(number_plane), FadeOut(hyperbolas), FadeOut(dyn_text))

        self.wait()

class EuclideanSymmetries(Scene):
    def construct(self):

        number_plane = NumberPlane(
            x_range=[-10, +10, 1],
            y_range=[-10, +10, 1]).set_z_index(-1)
        circle = Circle(radius = 1)
        stars = Mobject1D()
        n_stars = 12
        theta = 2 * np.pi / n_stars
        stars.add_points(
            [
                [np.cos(k * theta), np.sin(k * theta), 1]
                for k in range(n_stars)
            ]
        )

        #
        self.play(FadeIn(number_plane))
        self.wait(1)

        #
        self.play(ApplyComplexFunction(lambda x: x + 3 + 2*complex(0,1) , number_plane))
        self.wait(1)
        self.play(ApplyComplexFunction(lambda x: x - 3 - 2*complex(0,1) , number_plane))
        self.wait(1)

        #
        reflection = [[-1,0],[0,1]]
        triangle = Triangle(color = ORANGE).set_x(1).set_y(0).scale(0.1)
        self.play(FadeIn(circle, triangle))
        self.play(ApplyMatrix(reflection, number_plane), ApplyMatrix(reflection, circle), ApplyMatrix(reflection, triangle))
        self.wait(1)
        self.play(ApplyMatrix(reflection, number_plane), ApplyMatrix(reflection, circle), ApplyMatrix(reflection, triangle))
        self.wait(1)
        self.play(FadeOut(circle, triangle))

        #
        self.play(FadeIn(circle))
        self.add(stars)
        self.wait(1)
        self.play(FadeOut(number_plane))
        self.wait(1)

        #
        self.play(Rotate(stars, angle = 1/3 * PI))
        self.wait(1)

        #
        self.play(Rotate(stars, angle = -1/3 * PI))
        self.wait(1)

        #
        self.play(FadeIn(number_plane))
        self.play(Rotate(stars, angle = 1/3 * PI), Rotate(number_plane, angle = 1/3 * PI))
        self.wait(1)
        self.play(Rotate(stars, angle = -1/3 * PI), Rotate(number_plane, angle = -1/3 * PI))
        self.wait()

        #
        vector = Vector([3,2])
        vector_label = MathTex('(x,y)').set_x(3.2).set_y(2.2)
        self.remove(stars)
        self.play(FadeOut(circle), FadeIn(vector, vector_label))
        self.wait()

        #
        self.play(FadeOut(vector_label))
        theta = 2 * np.pi/3
        self.play(Rotate(vector, theta, about_point=ORIGIN))
        new_vector_label = MathTex('(x_R, y_R)').set_x(3.2 * np.cos(theta) - 2.2 * np.sin(theta) + 0.2).set_y(3.2 * np.sin(theta) + 2.2 * np.cos(theta) + 0.2)
        isometry_text = MathTex('x^2 + y^2 = x_R^2 + y_R^2', color = YELLOW).set_y(-2.5).set_x(-3)
        self.play(FadeIn(new_vector_label))
        self.wait()

        self.play(FadeIn(isometry_text))
        self.wait()
        self.play(FadeOut(number_plane, vector, isometry_text, new_vector_label))

class LorentzianSymmetries(Scene):
    def construct(self):

        number_plane = NumberPlane(
            x_range=[-10, +10, 1],
            y_range=[-10, +10, 1]).set_z_index(-1)
        upper_hyperbola = ParametricFunction(lambda xi: [np.sinh(xi), np.cosh(xi), 0], t_range = [-5, 5], color = BLUE)
        lower_hyperbola = ParametricFunction(lambda xi: [np.sinh(xi), -np.cosh(xi), 0], t_range = [-5, 5], color = BLUE)
        left_hyperbola = ParametricFunction(lambda xi: [-np.cosh(xi), np.sinh(xi), 0], t_range = [-5, 5], color = RED)
        right_hyperbola = ParametricFunction(lambda xi: [np.cosh(xi), np.sinh(xi), 0], t_range = [-5, 5], color = RED)
        hyperbolas = VGroup()
        hyperbolas.add(upper_hyperbola, lower_hyperbola, left_hyperbola, right_hyperbola)

        right_lightray = ParametricFunction(lambda xi: [xi, xi, 0], t_range = [-5, 5], color = PURPLE)
        left_lightray = ParametricFunction(lambda xi: [xi, -xi, 0], t_range = [-5, 5], color = PURPLE)
        lightrays = VGroup()
        lightrays.add(right_lightray, left_lightray)

        stars = Mobject1D()
        n = 6
        xi = 0.5
        for sign in [-1, +1]:
            stars.add_points(
                [
                    [np.sinh(k * xi), sign * np.cosh(k * xi), 1]
                    for k in np.arange(-n, n, 1)
                ]
            )
            stars.add_points(
                [
                    [sign * np.cosh(k * xi), np.sinh(k * xi), 1]
                    for k in np.arange(-n, n, 1)
                ]
            )
            stars.add_points(
                [
                    [k * xi, sign * k * xi, 1]
                    for k in np.arange(-n, n, 1)
                ]
            )
        
        self.play(FadeIn(hyperbolas, lightrays))
        self.add(stars)

        xi = 0.5
        boost_matrix = [[np.cosh(xi),np.sinh(xi)],[np.sinh(xi),np.cosh(xi)]]
        inv_boost_matrix = [[np.cosh(-xi),np.sinh(-xi)],[np.sinh(-xi),np.cosh(-xi)]]
        self.play(ApplyMatrix(boost_matrix, stars))
        self.wait()
        self.play(ApplyMatrix(inv_boost_matrix, stars))
        self.wait()

        self.play(FadeIn(number_plane))
        self.play(ApplyMatrix(boost_matrix, stars), ApplyMatrix(boost_matrix, number_plane))
        self.play(ApplyMatrix(inv_boost_matrix, stars), ApplyMatrix(inv_boost_matrix, number_plane))
        self.wait()

        self.remove(stars)
        self.play(FadeOut(lightrays, left_hyperbola, right_hyperbola))
        alice_vec = [np.sinh(-0.1), np.cosh(0.1)]
        vector = Vector(alice_vec)
        alice_vec_label = MathTex(r'(x_A, t_A)', color = YELLOW).set_x(alice_vec[0]- 2).set_y(alice_vec[1]-0.2)
        self.play(FadeIn(vector, alice_vec_label))
        self.wait()
        
        self.play(FadeOut(alice_vec_label))

        xi_2 = 0.2
        boost_matrix_2 = [[np.cosh(xi_2),np.sinh(xi_2)],[np.sinh(xi_2),np.cosh(xi_2)]]
        inv_boost_2 = [[np.cosh(xi_2),np.sinh(-xi_2)],[np.sinh(-xi_2),np.cosh(xi_2)]]
        self.play(ApplyMatrix(boost_matrix_2, vector), ApplyMatrix(boost_matrix_2, number_plane))
        bob_vec_label = MathTex(r'(x_B, t_B)', color = YELLOW).set_x(-2).set_y(np.cosh(0.5)-0.2)
        self.play(FadeIn(bob_vec_label))
        self.wait()
        self.play(FadeOut(bob_vec_label, vector))
        self.wait()

        self.play(ApplyMatrix(inv_boost_2, number_plane))
        self.wait()
        self.play(ApplyMatrix(inv_boost_2, number_plane))
        self.wait()
        self.play(FadeOut(number_plane, upper_hyperbola, lower_hyperbola))

class CelestialSphere0D(Scene):
    def construct(self):
        right_lightray = ParametricFunction(lambda xi: [xi, xi, 0], t_range = [-5, 5], color = PURPLE)
        left_lightray = ParametricFunction(lambda xi: [xi, -xi, 0], t_range = [-5, 5], color = PURPLE)
        lightrays = VGroup()
        lightrays.add(right_lightray, left_lightray)

        spacelike_slice = ParametricFunction(lambda x: [x, 1, 0], t_range = [-7, 7], color = GREEN)

        celestial_sphere = Mobject1D()
        celestial_sphere.add_points([[1,1,0], [-1,1,0]])

        self.play(FadeIn(lightrays))
        self.wait()
        self.play(FadeIn(spacelike_slice))
        self.wait()
        self.add(celestial_sphere)
        self.wait()

        xi = 0.5
        boost_matrix = [[np.cosh(xi),np.sinh(xi)],[np.sinh(xi),np.cosh(xi)]]
        self.play(ApplyMatrix(boost_matrix, celestial_sphere), ApplyMatrix(boost_matrix, spacelike_slice))
        self.wait()
        self.play(FadeOut(spacelike_slice))
        self.play(ApplyPointwiseFunction(lambda x: x/x[1], celestial_sphere))
        self.wait()
        self.remove(celestial_sphere)
        self.play(FadeOut(lightrays))

class BilinearForms(Scene):
    def construct(self):
        dyn_text = MathTex(r"x^2 + y^2 = \begin{pmatrix} x & y\end{pmatrix}\begin{pmatrix} x \\ y\end{pmatrix}") # metric in terms of vectors
        self.play(Write(dyn_text))

        euclidean_form = MathTex(r"x^2 + y^2 = \begin{pmatrix} x & y\end{pmatrix}\begin{pmatrix}1 & 0 \\ 0 & 1\end{pmatrix}\begin{pmatrix} x \\ y\end{pmatrix}")
        self.play(Transform(dyn_text, euclidean_form))

        lorentzian_form = MathTex(r"-t^2 + x^2 = \begin{pmatrix} t & x\end{pmatrix}\begin{pmatrix}-1 & 0 \\ 0 & 1\end{pmatrix}\begin{pmatrix} t \\ x\end{pmatrix}")
        self.play(Transform(dyn_text, lorentzian_form))

        self.play(Transform(dyn_text, euclidean_form))

        dyn_text_2 = MathTex(r"\begin{pmatrix}x \\ y \end{pmatrix} \mapsto R\begin{pmatrix}x \\ y \end{pmatrix}").set_y(-2) # mapsto for rotation
        self.play(Write(dyn_text_2))

        euclidean_mapsto = MathTex(r"x^2 + y^2 = \begin{pmatrix} x & y\end{pmatrix}R^T\begin{pmatrix}1 & 0 \\ 0 & 1\end{pmatrix}R\begin{pmatrix} x \\ y\end{pmatrix}")
        self.play(Transform(dyn_text, euclidean_mapsto))

        rotation_condition = MathTex(r"R^T\begin{pmatrix}1 & 0 \\ 0 & 1\end{pmatrix}R = \begin{pmatrix}1 & 0 \\ 0 & 1\end{pmatrix}")
        self.play(Transform(dyn_text, rotation_condition))

        rotation_condition_simp = MathTex(r"R^TR = \begin{pmatrix}1 & 0 \\ 0 & 1\end{pmatrix}, \det(R) = +1")
        self.play(Transform(dyn_text, rotation_condition_simp))

        rotation_theta = MathTex(r"R(\theta) = \begin{pmatrix}\cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta)\end{pmatrix}")
        self.play(Transform(dyn_text, rotation_theta))

        self.play(Transform(dyn_text, euclidean_mapsto))

        lorentz_tfmn = MathTex(r"\begin{pmatrix}t \\ x \end{pmatrix} \mapsto \Lambda\begin{pmatrix}t \\ x \end{pmatrix}").set_y(-2)
        lorentzian_mapsto = MathTex(r"-t^2 + x^2 = \begin{pmatrix} t & x\end{pmatrix}\Lambda^T\begin{pmatrix}-1 & 0 \\ 0 & 1\end{pmatrix}\Lambda\begin{pmatrix} t \\ x\end{pmatrix}")
        self.play(Transform(dyn_text_2, lorentz_tfmn), Transform(dyn_text, lorentzian_mapsto))

        lorentz_condition = MathTex(r"\Lambda^T\begin{pmatrix}-1 & 0 \\ 0 & 1\end{pmatrix}\Lambda = \begin{pmatrix}-1 & 0 \\ 0 & 1\end{pmatrix}")
        self.play(Transform(dyn_text, lorentz_condition))

        boost_xi = MathTex(r"\Lambda(\xi) = \begin{pmatrix}\cosh(\xi) & \sinh(\xi) \\ \sinh(\xi) & \cosh(\xi)\end{pmatrix}")
        self.play(Transform(dyn_text, boost_xi))

        self.wait()

class AliceAndBob(Scene):
    def construct(self):
        
        #
        number_plane = NumberPlane(
            x_range=[-10, +10, 1],
            y_range=[-10, +10, 1]).set_z_index(-1)

        vector = Vector([3, 2])

        vec_label = MathTex(r'(x,t)').set_x(3.2).set_y(2.2)

        self.play(FadeIn(number_plane, vector, vec_label))
        self.wait(1)

        #
        dyn_text = MathTex(r'\Delta s^2 := \text{Pseudo-distance} = -t^2 + x^2', color = YELLOW).set_x(-3).set_y(-2.5)

        self.play(FadeIn(dyn_text))
        self.wait(1)

        #
        alice_text = MathTex(r'\text{Alice: The event is at }(x_A,t_A)', color = YELLOW).set_x(-3).set_y(-2.5)
        alice_vec = Vector([np.sinh(-1), np.cosh(-1)])
        self.play(FadeOut(vec_label), Transform(dyn_text, alice_text), Transform(vector, alice_vec))
        self.wait(1)

        #
        bob_text = MathTex(r'\text{Bob: No! The event is at }(x_B,t_B)', color = YELLOW).set_x(-3).set_y(-2.5)
        bob_vec = Vector([np.sinh(2), np.cosh(2)])
        self.play(Transform(dyn_text, bob_text), Transform(vector, bob_vec))
        self.wait(1)

        #
        agree_text = MathTex(r'\Delta s^2 = -t_A^2 + x_A^2 = -t_B^2 + x_B^2', color = YELLOW).set_x(-3).set_y(-2.5)
        self.play(Transform(dyn_text, agree_text))
        self.wait(1)

        #
        self.play(FadeOut(dyn_text, number_plane, vector))




