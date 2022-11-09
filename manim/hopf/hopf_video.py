###
# hopf.py
#
# Contains manim Scenes for hopf fibration video
###

'''
Dependencies
'''

from ast import Param
from manim import *
from hopf_utils import *
from lie_utils import *
import numpy as np

'''
Scenes
'''

class PreimageTest(ThreeDScene):
    def construct(self):

        # Camera
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        # Mobjects
        n = np.array([1,0,0])
        n_preimage = ParametricFunction(S2_to_preimage(n))

        # Animations
        self.add(n_preimage)

class HomotopyTest(ThreeDScene):
    def construct(self):
        # Camera
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        # Mobjects
        n = np.array([0,1,0])
        n_preimage = ParametricFunction(S2_to_preimage(n)).set_color(PURPLE)

        axes = ThreeDAxes(x_range = [0,1,1], y_range = [0,1,1], z_range = [0,1,1], x_length=3, y_length = 3, z_length = 3).set_x(1.42).set_y(1.42).set_z(1.5)

        # Animations
        self.add(n_preimage, axes)
        self.wait()
        
        turn_animation_into_updater(FadeToColor(n_preimage, BLUE))
        self.play(Homotopy(lambda x,y,z,t: pauli_rot_homotopy(x,y,z,-t/2), n_preimage), rate_function=linear) # Pauli rotation homotopy
        # self.play(Homotopy(apollonian_tori_homotopy, n_preimage), rate_function=linear) # Apollonian tori homotopy, less stable
        self.wait()

class VillarceauTorusTest(ThreeDScene):
    def construct(self):
        # Camera
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        # Mobjects
        n = np.array([0,1,0])
        torus = Surface(lambda u,v: villarceau_torus(v,u,n), resolution=(24,24)).set_fill_by_checkerboard(PURPLE, PURPLE)

        axes = ThreeDAxes(x_range = [0,1,1], y_range = [0,1,1], z_range = [0,1,1], x_length=3, y_length = 3, z_length = 3).set_x(1.42).set_y(1.42).set_z(1.5)

        # Animations
        self.add(torus, axes)
        self.wait()
        turn_animation_into_updater(FadeToColor(torus, RED))
        self.play(Homotopy(lambda x,y,z,t: apollonian_tori_homotopy(x,y,z,t/4), torus))
        self.wait()

class DrawVillarceauTorus(ThreeDScene):
    def construct(self):
        # Camera
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        # Mobjects
        n = np.array([0,1,0])
        circle = ParametricFunction(S2_to_preimage(n)).set_color(PURPLE)
        torus = Surface(lambda u,v: villarceau_torus(v,u,n), resolution=(24,24)).set_fill_by_checkerboard(PURPLE, PURPLE)

        axes = ThreeDAxes(x_range = [0,1,1], y_range = [0,1,1], z_range = [0,1,1], x_length=3, y_length = 3, z_length = 3).set_x(1.42).set_y(1.42).set_z(1.5)

        # Animations
        self.add(axes)
        self.play(Create(circle))
        self.wait()
        self.play(Create(torus), FadeOut(circle))
        self.wait()

class FlashAxes(ThreeDScene):
    def construct(self):
        # Camera
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        # Mobjects
        axes = ThreeDAxes(x_range = [0,1,1], y_range = [0,1,1], z_range = [0,1,1], x_length=3, y_length = 3, z_length = 3).set_x(0.42).set_y(0.42).set_z(0.5)

        # Animations
        self.play(FadeIn(axes))
        self.wait()
        self.play(FadeOut(axes))
        self.wait()

class DrawAxesAndSphere(ThreeDScene):
    def construct(self):
        # Camera
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        # Mobjects
        axes = ThreeDAxes(x_range = [0,1,1], y_range = [0,1,1], z_range = [0,1,1], x_length=1, y_length = 1, z_length = 1).set_x(0.42).set_y(0.42).set_z(0.5)
        sphere = Surface(lambda u,v: np.array([np.cos(u * 2 * np.pi) * np.sin(v * np.pi), np.sin(u * 2 * np.pi) * np.sin(v * np.pi), np.cos(v * np.pi)]), 
                         resolution=(12,12), 
                         fill_opacity=0.5).set_fill_by_checkerboard(GREY, GREY)

        # Animations
        self.play(FadeIn(axes, sphere))
        self.wait()
        self.play(FadeOut(axes, sphere))
        self.wait()

class RotateUnitVector(ThreeDScene):
    def construct(self):
        # Camera
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(3)

        # Mobjects
        axes = ThreeDAxes(x_range = [0,1,1], y_range = [0,1,1], z_range = [0,1,1], x_length=1, y_length = 1, z_length = 1).set_x(0.42).set_y(0.42).set_z(0.5)
        sphere = Surface(lambda u,v: np.array([np.cos(u * 2 * np.pi) * np.sin(v * np.pi), np.sin(u * 2 * np.pi) * np.sin(v * np.pi), np.cos(v * np.pi)]), 
                         resolution=(12,12), 
                         fill_opacity=0.5).set_fill_by_checkerboard(GREY, GREY)

        unit_pt = Dot3D(color = PURPLE).set_x(0).set_y(1).set_z(0)

        # Functions for animation
        rotate_about_x = lambda x,y,z,t: np.dot(expm(-t/2 * np.pi * np.array([[0,0,0],[0,0,-1],[0,1,0]])), np.array([x,y,z]))

        # Animations
        self.add(axes,sphere)
        self.play(FadeIn(unit_pt))
        self.wait()
        turn_animation_into_updater(FadeToColor(unit_pt, BLUE))
        self.play(Homotopy(rotate_about_x, unit_pt))
        self.wait()

class UnitVectorToCircle(ThreeDScene):
    def construct(self):
        # Camera
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(3)

        # Mobjects
        axes = ThreeDAxes(x_range = [0,1,1], y_range = [0,1,1], z_range = [0,1,1], x_length=1, y_length = 1, z_length = 1).set_x(0.42).set_y(0.42).set_z(0.5)
        sphere = Surface(lambda u,v: np.array([np.cos(u * 2 * np.pi) * np.sin(v * np.pi), np.sin(u * 2 * np.pi) * np.sin(v * np.pi), np.cos(v * np.pi)]), 
                         resolution=(12,12), 
                         fill_opacity=0.5).set_fill_by_checkerboard(GREY, GREY)

        unit_pt = Dot3D(color = PURPLE).set_x(0).set_y(1).set_z(0)
        circle = ParametricFunction(lambda t: np.array([-np.sin(2 * np.pi * t), np.cos(2 * np.pi * t), 0]), color = PURPLE)

        # Animations
        self.add(axes,sphere, unit_pt)
        self.wait()
        self.play(Create(circle), FadeOut(unit_pt))
        self.wait()
        
class SphereCircleHomotopyTest(ThreeDScene):
    def construct(self):
        # Camera
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(3)

        # Mobjects
        axes = ThreeDAxes(x_range = [0,1,1], y_range = [0,1,1], z_range = [0,1,1], x_length=1, y_length = 1, z_length = 1).set_x(0.42).set_y(0.42).set_z(0.5)
        sphere = Surface(lambda u,v: np.array([np.cos(u * 2 * np.pi) * np.sin(v * np.pi), np.sin(u * 2 * np.pi) * np.sin(v * np.pi), np.cos(v * np.pi)]), 
                         resolution=(12,12), 
                         fill_opacity=0.5).set_fill_by_checkerboard(GREY, GREY)

        circle = ParametricFunction(lambda t: np.array([-np.sin(2 * np.pi * t), np.cos(2 * np.pi * t), 0]), color = PURPLE)

        # Animations
        self.add(axes, sphere, circle)
        self.wait()
        turn_animation_into_updater(FadeToColor(circle, BLUE))
        self.play(Homotopy(lambda x,y,z,t: z_boost_homotopy(x,y,z,-t), circle))

class NestedTori(ThreeDScene):
    def construct(self):
        # Camera
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        # Mobjects
        axes = ThreeDAxes(x_range = [0,1,1], y_range = [0,1,1], z_range = [0,1,1], x_length=3, y_length = 3, z_length = 3).set_x(1.42).set_y(1.42).set_z(1.5)

        inner_torus = Surface(lambda u,v: torus_min_maj(u,v,r=3/4, R = 5/4),
                              u_range = [0, 2 * np.pi],
                              v_range = [0, 2 * np.pi],
                              resolution = (16,16),
                              fill_opacity = 0.5).set_fill_by_checkerboard(RED,RED)

        mid_torus = Surface(lambda u,v: torus_min_maj(u,v,r=4/3, R = 5/3),
                              u_range = [0, 2 * np.pi],
                              v_range = [0, 2 * np.pi],
                              resolution = (16,16),
                              fill_opacity = 0.5).set_fill_by_checkerboard(PURPLE,PURPLE)

        outer_torus = Surface(lambda u,v: torus_min_maj(u,v,r=12/5, R = 13/5),
                              u_range = [0, 2 * np.pi],
                              v_range = [0, 2 * np.pi],
                              resolution = (16,16),
                              fill_opacity = 0.5).set_fill_by_checkerboard(BLUE,BLUE)

        # Animations
        self.play(FadeIn(axes))
        self.wait()
        self.play(FadeIn(inner_torus))
        self.wait()
        self.play(FadeIn(mid_torus))
        self.wait()
        self.play(FadeIn(outer_torus))
        self.wait()
        self.play(FadeOut(axes, inner_torus, mid_torus, outer_torus))

class TorusSlice(ThreeDScene):
    def construct(self):
        # Camera
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        # Mobjects
        axes = ThreeDAxes(x_range = [0,1,1], y_range = [0,1,1], z_range = [0,1,1], x_length=3, y_length = 3, z_length = 3).set_x(1.42).set_y(1.42).set_z(1.5)
        plane_slice = Surface(lambda u,v: 5 * np.array([(2*u-1),2 * v - 1,0]), resolution = (16,16), fill_opacity=0.5).set_fill_by_checkerboard(GREEN,GREEN)
        torus = Surface(lambda u,v: 5/4 * np.array([np.cos(u), np.sin(u),0]) + np.cos(v) * np.array([np.cos(u), np.sin(u), 0]) + np.sin(v) * np.array([0,0,1]), 
                u_range = [0,2 * np.pi], 
                v_range = [0,2 * np.pi],
                resolution = (16,16)).set_fill_by_checkerboard(PURPLE,PURPLE)

        upper_torus = Surface(lambda u,v: 5/4 * np.array([np.cos(u), np.sin(u),0]) + 3/4 *( np.cos(v) * np.array([np.cos(u), np.sin(u), 0]) + np.sin(v) * np.array([0,0,1])), 
                u_range = [0,2 * np.pi], 
                v_range = [0,np.pi],
                resolution = (16,8),
                fill_opacity=0.5).set_fill_by_checkerboard(PURPLE,PURPLE)

        lower_torus = Surface(lambda u,v: 5/4 * np.array([np.cos(u), np.sin(u),0]) + 3/4 *( np.cos(v) * np.array([np.cos(u), np.sin(u), 0]) + np.sin(v) * np.array([0,0,1])), 
                u_range = [0,2 * np.pi], 
                v_range = [-np.pi,0],
                resolution = (16,8),
                fill_opacity=0.5).set_fill_by_checkerboard(PURPLE,PURPLE)

        circles = VGroup()
        for n in range(5):
            outer_circle = Circle(radius = n+1).set_color(WHITE)
            inner_circle = Circle(radius = 1/(n+1)).set_color(WHITE)
            circles.add(outer_circle, inner_circle)

        mid_circles = VGroup()
        mid_circles.add(Circle(1/2).set_color(PURPLE), Circle(2).set_color(PURPLE))

        top_circle = Circle(1).set_color(RED)

        bot_circle = Dot().set_color(BLUE)

        # Animations
        self.add(axes)
        self.play(FadeIn(upper_torus, lower_torus))
        self.wait()
        self.play(FadeOut(upper_torus), FadeIn(plane_slice))
        self.wait()
        self.begin_ambient_camera_rotation(rate=-75 * DEGREES, about="phi")
        self.wait()
        self.stop_ambient_camera_rotation(about="phi")
        self.play(FadeIn(mid_circles))
        self.wait()
        self.play(FadeIn(top_circle))
        self.wait()
        self.play(FadeIn(bot_circle))
        self.wait()
        self.play(FadeIn(circles))
        self.wait()

class TorusVertSlice(ThreeDScene):
    def construct(self):
        # Camera
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        # Mobjects
        axes = ThreeDAxes(x_range = [0,1,1], y_range = [0,1,1], z_range = [0,1,1], x_length=3, y_length = 3, z_length = 3).set_x(1.42).set_y(1.42).set_z(1.5)
        plane_slice = Surface(lambda u,v: 5 * np.array([0,2*u-1, 2*v - 1]), resolution = (16,16), fill_opacity=0.5).set_fill_by_checkerboard(GREEN,GREEN)
        torus = Surface(lambda u,v: 5/4 * np.array([np.cos(u), np.sin(u),0]) + np.cos(v) * np.array([np.cos(u), np.sin(u), 0]) + np.sin(v) * np.array([0,0,1]), 
                u_range = [0,2 * np.pi], 
                v_range = [0,2 * np.pi],
                resolution = (16,16)).set_fill_by_checkerboard(PURPLE,PURPLE)

        front_torus = Surface(lambda v,u: 5/4 * np.array([np.cos(u), np.sin(u),0]) + 3/4 *( np.cos(v) * np.array([np.cos(u), np.sin(u), 0]) + np.sin(v) * np.array([0,0,1])), 
                u_range = [0,2 * np.pi], 
                v_range = [-np.pi/2,+np.pi/2],
                resolution = (16,8),
                fill_opacity=0.5).set_fill_by_checkerboard(PURPLE,PURPLE)

        back_torus = Surface(lambda v,u: 5/4 * np.array([np.cos(u), np.sin(u),0]) + 3/4 *( np.cos(v) * np.array([np.cos(u), np.sin(u), 0]) + np.sin(v) * np.array([0,0,1])), 
                u_range = [0,2 * np.pi], 
                v_range = [+np.pi/2,3 * np.pi/2],
                resolution = (16,8),
                fill_opacity=0.5).set_fill_by_checkerboard(PURPLE,PURPLE)

        circles = VGroup()
        for n in range(5):
            d = (n+1)/(5+1)
            R = (1 + d**2)/(1 - d**2)
            r = 2*d/(1 - d**2)
            left_circle = ParametricFunction(lambda t: np.array([0, -R + r * np.cos(t), r * np.sin(t)]), t_range = [0,2 * np.pi])
            right_circle = ParametricFunction(lambda t: np.array([0, +R + r * np.cos(t), r * np.sin(t)]), t_range = [0,2 * np.pi])
            circles.add(left_circle, right_circle)

        mid_circles = VGroup()
        mid_circles.add(ParametricFunction(lambda t: np.array([0, -5/4 + 3/4 * np.cos(t), 3/4 * np.sin(t)]), t_range = [0,2 * np.pi]).set_color(PURPLE),
            ParametricFunction(lambda t: np.array([0, +5/4 + 3/4 * np.cos(t), 3/4 * np.sin(t)]), t_range = [0,2 * np.pi]).set_color(PURPLE))

        top_circles = VGroup()
        top_circles.add(Dot3D(color = RED).set_y(-1), Dot3D(color = RED).set_y(+1))

        bot_circle = ParametricFunction(lambda t: np.array([0,0, t]), t_range = [-5, 5]).set_color(BLUE)

        # Animations
        self.add(axes)
        self.play(FadeIn(front_torus, back_torus))
        self.wait()
        self.play(FadeOut(front_torus), FadeIn(plane_slice))
        self.wait()
        self.begin_ambient_camera_rotation(rate=+15 * DEGREES, about="phi")
        self.wait()
        self.stop_ambient_camera_rotation(about="phi")
        self.play(FadeIn(mid_circles))
        self.wait()
        self.play(FadeIn(top_circles))
        self.wait()
        self.play(FadeIn(bot_circle))
        self.wait()
        self.play(FadeIn(circles))
        self.wait()

class VillarceauCut(ThreeDScene):
    def construct(self):
        # Camera
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        # Mobjects
        axes = ThreeDAxes(x_range = [0,1,1], y_range = [0,1,1], z_range = [0,1,1], x_length=3, y_length = 3, z_length = 3).set_x(1.42).set_y(1.42).set_z(1.5)
        torus = Surface(lambda u,v:  np.array([(2+ np.cos(v)) * np.cos(u), (2+ np.cos(v)) * np.sin(u), np.sin(v)]),
                        u_range = [0, 2 * np.pi],
                        v_range = [0, 2 * np.pi],
                        resolution = (16,16),
                        fill_opacity=0.5).set_fill_by_checkerboard(PURPLE,PURPLE)
        back_half_torus = Surface(lambda u,v:  np.array([(2+ np.cos(v)) * np.cos(u), (2+ np.cos(v)) * np.sin(u), np.sin(v)]),
                        u_range = [np.pi/2, 3 * np.pi/2],
                        v_range = [0, 2 * np.pi],
                        resolution = (16,16),
                        fill_opacity=0.5).set_fill_by_checkerboard(PURPLE,PURPLE)

        front_half_torus = Surface(lambda u,v:  np.array([(2+ np.cos(v)) * np.cos(u), (2+ np.cos(v)) * np.sin(u), np.sin(v)]),
                        u_range = [-np.pi/2, np.pi/2],
                        v_range = [0, 2 * np.pi],
                        resolution = (16,16),
                        fill_opacity=0.5).set_fill_by_checkerboard(PURPLE,PURPLE)

        vertical_plane = Surface(lambda u,v: np.array([0,u,v]),
                                 u_range = [-4,4],
                                 v_range = [-4,4],
                                 resolution = (2,2),
                                 fill_opacity = 0.5).set_fill_by_checkerboard(ORANGE, ORANGE)

        intersection_circles = VGroup()
        intersection_circle_1 = ParametricFunction(lambda t: np.array([0, 2 + np.cos(t), np.sin(t)]), 
                                                   t_range = [0, 2 * np.pi],
                                                   color = RED)
        intersection_circle_2 = ParametricFunction(lambda t: np.array([0, -2 + np.cos(t), np.sin(t)]), 
                                                   t_range = [0, 2 * np.pi],
                                                   color = RED)
        intersection_circles.add(intersection_circle_1, intersection_circle_2)

        tangent_line = ParametricFunction(lambda t: np.array([0, t, t/np.sqrt(3)]),
                                          t_range = [-4, 4],
                                          color = BLUE)

        normal = Vector(np.array([0, -1/2, np.sqrt(3)/2]), color = GREEN)

        villarceau_plane = Surface(lambda u,v: np.array([u, v, v / np.sqrt(3)]),
                                   u_range = [-4, 4],
                                   v_range = [-4, 4],
                                   resolution = (2,2),
                                   fill_opacity = 0.5).set_fill_by_checkerboard(BLUE, BLUE)

        partial_villarceau_circles = VGroup()

        partial_villarceau_circle_1 = ParametricFunction(lambda t: np.array([1 + 2 * np.sin(t), np.sqrt(3) * np.cos(t), np.cos(t)]), 
                                                 t_range = [-5 * np.pi/6, -np.pi/6])
        partial_villarceau_circle_2 = ParametricFunction(lambda t: np.array([ -1 + 2 * np.sin(t), np.sqrt(3) * np.cos(t), np.cos(t)]), 
                                                 t_range = [-7 * np.pi/6, np.pi/6])
        partial_villarceau_circles.add(partial_villarceau_circle_1, partial_villarceau_circle_2)

        villarceau_circles = VGroup()

        villarceau_circle_1 = ParametricFunction(lambda t: np.array([1 + 2 * np.sin(t), np.sqrt(3) * np.cos(t), np.cos(t)]), 
                                                 t_range = [0, 2 * np.pi],
                                                 )
        villarceau_circle_2 = ParametricFunction(lambda t: np.array([ -1 + 2 * np.sin(t), np.sqrt(3) * np.cos(t), np.cos(t)]), 
                                                 t_range = [0, 2 * np.pi],
                                                 )
        villarceau_circles.add(villarceau_circle_1, villarceau_circle_2)

        villarceau_cutaway_surface = Surface(villarceau_cutaway,
                                             u_range = [-2 * np.pi, 2 * np.pi],
                                             v_range = [0, 2 * np.pi],
                                             resolution = (16,16),
                                             fill_opacity = 0.5).set_fill_by_checkerboard(PURPLE, PURPLE)


        # Animations
        self.play(FadeIn(axes))
        self.wait()
        self.play(FadeIn(front_half_torus, back_half_torus))
        self.wait()
        self.play(FadeIn(vertical_plane), FadeOut(front_half_torus))
        self.wait()
        self.play(FadeIn(intersection_circles))
        self.wait()
        self.begin_ambient_camera_rotation(rate = 15 * DEGREES, about = "phi")
        self.wait()
        self.stop_ambient_camera_rotation(about="phi")
        self.wait()
        self.play(FadeIn(tangent_line))
        self.wait()
        self.play(FadeIn(normal))
        self.wait()
        self.play(FadeOut(vertical_plane, intersection_circles, tangent_line), FadeIn(villarceau_plane, partial_villarceau_circles))
        self.wait()
        self.begin_ambient_camera_rotation(rate = -np.pi/2, about = "theta")
        self.wait()
        self.stop_ambient_camera_rotation(about = "theta")
        self.wait()
        self.play(FadeIn(front_half_torus, villarceau_circles), FadeOut(partial_villarceau_circles))
        self.wait()
        self.play(FadeIn(villarceau_cutaway_surface), FadeOut(front_half_torus, back_half_torus, normal))
        self.wait()
        self.play(FadeOut(villarceau_plane))
        self.wait()
        self.begin_ambient_camera_rotation(rate = np.pi/2, about = "theta")
        self.wait(4.05)
        self.stop_ambient_camera_rotation(about = "theta")
        self.play(FadeOut(villarceau_cutaway_surface, villarceau_circles), FadeIn(villarceau_circle_1))

class RotateVillarceauCircle(ThreeDScene):
    def construct(self):
        # Camera
        phi = 90
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        # Mobjects
        axes = ThreeDAxes(x_range = [0,1,1], y_range = [0,1,1], z_range = [0,1,1], x_length=3, y_length = 3, z_length = 3).set_x(1.42).set_y(1.42).set_z(1.5)
        
        villarceau_circle_1 = ParametricFunction(lambda t: np.array([1 + 2 * np.sin(t), np.sqrt(3) * np.cos(t), np.cos(t)]), 
                                                 t_range = [0, 2 * np.pi],
                                                 )

        villarceau_torus = Surface(lambda u,v: np.dot(expm(u * np.array([[0,-1,0], [1,0,0], [0,0,0]])),
                                                      np.array([(1 + 2 * np.sin(v)), (np.sqrt(3) * np.cos(v)), np.cos(v)])),
                                   u_range = [0, 2 * np.pi],
                                   v_range = [0, 2 * np.pi],
                                   resolution = (16,16),
                                   fill_opacity = 0.5).set_fill_by_checkerboard(PURPLE, PURPLE)

        # Animations
        self.play(FadeIn(axes, villarceau_circle_1))
        self.begin_ambient_camera_rotation(rate = -15 * DEGREES, about = "phi")
        self.wait()
        self.stop_ambient_camera_rotation(about = "phi")
        self.wait()
        self.play(Create(villarceau_torus), Homotopy(lambda x,y,z,t: np.dot(expm(2 * np.pi * t * np.array([[0,-1,0], [1,0,0], [0,0,0]])), np.array([x,y,z])), 
                                                     villarceau_circle_1, rate_function = "linear"), )
        self.wait()
        self.play(FadeOut(villarceau_circle_1))
        self.wait()
        self.play(FadeOut(villarceau_torus, axes))

class VillarceauCutawayTest(ThreeDScene):
    def construct(self):
        # Camera
        phi = 0
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        # Mobjects

        villarceau_cutaway_surface = Surface(villarceau_cutaway,
                                             u_range = [-2*np.pi, 2 * np.pi],
                                             v_range = [0, 2 * np.pi],
                                             resolution = (16,16),
                                             fill_opacity = 0.5).set_fill_by_checkerboard(GREEN, GREEN)
        villarceau = VGroup()
        villarceau.add(villarceau_cutaway_surface,)

        # Animations
        self.add(villarceau)
        self.begin_ambient_camera_rotation(rate = np.pi, about = "phi")
        self.wait(2)
        self.stop_ambient_camera_rotation(about = "phi")
        self.begin_ambient_camera_rotation(rate = np.pi, about = "theta")
        self.wait(2)
        self.stop_ambient_camera_rotation(about = "theta")

class ApollonianProjection(ThreeDScene):
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
                            
        orbits = []
        thetas = np.arange(0, np.pi, np.pi/12)
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
        
        text = Tex('Apollonian circles').set_y(1.5).set_color(YELLOW)
        
        # Animations

        self.play(FadeIn(punctured_sphere))

        self.play( 
            FadeIn(*orbits)
            )
        self.wait()

        self.play(
            ApplyPointwiseFunction(stereographic_projection_3d, punctured_sphere),
            ApplyPointwiseFunction(stereographic_projection_3d, orbits[0]),
            ApplyPointwiseFunction(stereographic_projection_3d, orbits[1]),
            ApplyPointwiseFunction(stereographic_projection_3d, orbits[2]),
            ApplyPointwiseFunction(stereographic_projection_3d, orbits[3]),
            ApplyPointwiseFunction(stereographic_projection_3d, orbits[4]),
            ApplyPointwiseFunction(stereographic_projection_3d, orbits[5]),
            ApplyPointwiseFunction(stereographic_projection_3d, orbits[6]),
            ApplyPointwiseFunction(stereographic_projection_3d, orbits[7]),
            ApplyPointwiseFunction(stereographic_projection_3d, orbits[8]),
            ApplyPointwiseFunction(stereographic_projection_3d, orbits[9]),
            ApplyPointwiseFunction(stereographic_projection_3d, orbits[10]),
            ApplyPointwiseFunction(stereographic_projection_3d, orbits[11]),    
            )
        self.wait()

        self.begin_ambient_camera_rotation(rate = -75*DEGREES, about = 'phi')
        self.wait()
        self.stop_ambient_camera_rotation(about = 'phi')
        self.wait()
        self.play(FadeIn(text))

class StereographicLine(ThreeDScene):
    def construct(self):
        # Camera
        phi = 75
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(2)

        # Mobjects
        axes = ThreeDAxes(x_range = [0,1,1], y_range = [0,1,1], z_range = [0,1,1], x_length=1, y_length = 1, z_length = 1).set_x(.42).set_y(.42).set_z(.5)

        punctured_sphere = Surface(lambda u,v: np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)]),
                               u_range = [0, 2 * np.pi],
                               v_range = [np.pi/12, np.pi],
                               resolution = (12, 11),
                               fill_opacity = 0.7).set_fill_by_checkerboard(PURPLE, PURPLE)

        stereog_line = ParametricFunction(lambda t: np.array([t, 0, 0]), t_range = [-1/np.tan(np.pi/24), +1/np.tan(np.pi/24)]).set_color(GREEN)
        
        # Animations
        self.play(FadeIn(axes))
        self.wait()
        self.play(FadeIn(punctured_sphere))
        self.wait()
        self.play(ApplyPointwiseFunction(stereographic_projection_3d, punctured_sphere))
        self.wait()
        self.begin_ambient_camera_rotation(rate = -75*DEGREES, about = 'phi')
        self.wait()
        self.stop_ambient_camera_rotation(about = 'phi')
        self.wait()
        self.play(FadeIn(stereog_line))
        self.wait()
        self.begin_ambient_camera_rotation(rate = +75*DEGREES, about = 'phi')
        self.wait()
        self.stop_ambient_camera_rotation(about = 'phi')
        self.wait()
        self.play(ApplyPointwiseFunction(stereographic_inverse_3d, punctured_sphere), 
                  ApplyPointwiseFunction(stereographic_inverse_3d, stereog_line))

class GaugeTransformation(ThreeDScene):
    def construct(self):
        # Camera
        phi = 75
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(2)

        # Mobjects
        axes = ThreeDAxes(x_range = [0,1,1], y_range = [0,1,1], z_range = [0,1,1], x_length=1, y_length = 1, z_length = 1).set_x(.42).set_y(.42).set_z(.5)

        disc = Surface(lambda u,v: u * np.array([np.cos(v), np.sin(v), 0]),
                       u_range = [0,1],
                       v_range = [0, 2 * np.pi],
                       resolution = (12,12),
                       fill_opacity = 0.5).set_fill_by_checkerboard(ORANGE, ORANGE)
        
        # Animations
        self.play(FadeIn(axes))
        self.wait()
        self.play(FadeIn(disc))
        self.wait()
        self.play(Homotopy(funky_fibrewise_rot_homotopy, disc))
        self.wait()