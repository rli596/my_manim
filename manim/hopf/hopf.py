###
# hopf.py
#
# Contains manim Scenes for hopf map
###

'''
Dependencies
'''

from manim import *
from utils import *
from lie_utils import *
import numpy as np

'''
Scenes
'''

# Stationary scenes
class S_2(ThreeDScene):
    def construct(self):
        #Omega = (np.pi/3, np.pi/6)
        #n = angle_to_3_vec(Omega)
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        thetas = [np.pi/2 -np.pi/3, np.pi/2 , np.pi/2 + np.pi/3]
        phis = np.linspace(0, 2 * np.pi, 100)
        Omegas = []
        for theta in thetas:
            for phi in phis:
                Omegas.append([theta, phi])
        Omegas = np.array(Omegas)

        pts = VGroup()
        for Omega in Omegas:
            n = angle_to_3_vec(Omega)

            theta = Omega[0]
            if theta < np.pi/2:
                colour = RED
            if theta == np.pi/2:
                colour = WHITE
            if theta > np.pi/2:
                colour = BLUE
            pts += Dot(n, color = colour)
        self.add(pts)

class S2Spiral(ThreeDScene):
    def construct(self):
        #Omega = (np.pi/3, np.pi/6)
        #n = angle_to_3_vec(Omega)
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        thetas = np.linspace(0, 2/3*np.pi, 1000)
        Omegas = []
        n = 10
        for theta in thetas:
            Omegas.append([theta, 2*n*theta])
        Omegas = np.array(Omegas)

        pts = VGroup()
        for Omega in Omegas:
            n = angle_to_3_vec(Omega)

            theta = Omega[0]
            colour = BLUE
            pts += Dot(n, color = colour)
        self.add(pts)

class StereogReps(ThreeDScene):
    def construct(self):
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        thetas = np.linspace(0, np.pi, 10)
        phis = np.linspace(0, 2 * np.pi, 20)
        Omegas = []
        for theta in thetas:
            for phi in phis:
                Omegas.append([theta, phi])
        Omegas = np.array(Omegas)
        pts = VGroup()
        for Omega in Omegas:
            x = angle_to_rep_4_vec(Omega)
            X = stereog_proj(x)
            pts += Dot(X)
        self.add(pts)

class TwoBallReps(ThreeDScene):
    def construct(self):
        phi = 75
        theta = 90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        thetas = np.linspace(0, np.pi, 10)
        phis = np.linspace(0, 2 * np.pi, 20)
        Omegas = []
        for theta in thetas:
            for phi in phis:
                Omegas.append([theta, phi])
        Omegas = np.array(Omegas)
        pts = VGroup()
        for Omega in Omegas:
            x = angle_to_rep_4_vec(Omega)
            X = two_balls(x)
            pts += Dot(X)
        self.add(pts)

class StereogOrbit(ThreeDScene):
    def construct(self):
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        thetas = [-np.pi/3]
        phis = np.linspace(0, 2 * np.pi, 20)
        phi_dashes = np.linspace(0, 4 * np.pi, 20)
        Omegas = []
        for theta in thetas:
            for phi in phis:
                Omegas.append([theta, phi])
        Omegas = np.array(Omegas)
        pts = VGroup()
        for Omega in Omegas:
            for phi_dash in phi_dashes:
                x = angle_to_rep_4_vec(Omega)
                x_dash = rep_4_vec_orbit(x, phi_dash)
                X = stereog_proj(x_dash)
                pts += Dot(X)
        self.add(pts)

class TwoBallsOrbit(ThreeDScene):
    def construct(self):
        phi = 75
        theta = 90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        thetas = [-np.pi/3]
        phis = np.linspace(0, 2 * np.pi, 20)
        phi_dashes = np.linspace(0, 4 * np.pi, 20)
        Omegas = []
        for theta in thetas:
            for phi in phis:
                Omegas.append([theta, phi])
        Omegas = np.array(Omegas)
        pts = VGroup()
        for Omega in Omegas:
            for phi_dash in phi_dashes:
                x = angle_to_rep_4_vec(Omega)
                z = real_4_to_cx_2(x)
                z_0 = [z[0]*np.exp(complex(0, phi_dash/2)), z[1]*np.exp(complex(0, 0))]
                x_0 = cx_2_to_real_4(z_0)
                z_1 = [z[0]*np.exp(complex(0, 0)), z[1]*np.exp(complex(0, phi_dash/2))]
                x_1 = cx_2_to_real_4(z_1)
                X_0 = stereog_proj(x_0)
                X_1 = stereog_proj(x_1)
                pts += Dot(X_0, color = BLUE)
                pts += Dot(X_1, color = RED)
        self.add(pts)

class StereogOrbitConstz_1(ThreeDScene):
    def construct(self):
        phi = 150
        theta = 90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        thetas = [-np.pi/3]
        phis = [np.pi/6]
        phi_dashes = np.linspace(0, 4 * np.pi, 20)
        Omegas = []
        for theta in thetas:
            for phi in phis:
                Omegas.append([theta, phi])
        Omegas = np.array(Omegas)
        pts = VGroup()
        for Omega in Omegas:
            for phi_dash in phi_dashes:
                x = angle_to_rep_4_vec(Omega)
                z = real_4_to_cx_2(x)
                z_0 = [z[0]*np.exp(complex(0, phi_dash/2)), z[1]*np.exp(complex(0, 0))]
                x_0 = cx_2_to_real_4(z_0)
                z_1 = [z[0]*np.exp(complex(0, 0)), z[1]*np.exp(complex(0, phi_dash/2))]
                x_1 = cx_2_to_real_4(z_1)
                X_0 = stereog_proj(x_0)
                X_1 = stereog_proj(x_1)
                pts += Dot(X_0, color = BLUE)
                pts += Dot(X_1, color = RED)
        self.add(pts)

class TwoBallsOrbitConstz_1(ThreeDScene):
    def construct(self):
        phi = 30
        theta = 90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        thetas = [np.pi/2]
        phis = [np.pi/6]
        phi_dashes = np.linspace(0, 4 * np.pi, 20)
        Omegas = []
        for theta in thetas:
            for phi in phis:
                Omegas.append([theta, phi])
        Omegas = np.array(Omegas)
        pts = VGroup()
        for Omega in Omegas:
            for phi_dash in phi_dashes:
                x = angle_to_rep_4_vec(Omega)
                z = real_4_to_cx_2(x)
                z_0 = [z[0]*np.exp(complex(0, phi_dash/2)), z[1]*np.exp(complex(0, 0))]
                x_0 = cx_2_to_real_4(z_0)
                z_1 = [z[0]*np.exp(complex(0, 0)), z[1]*np.exp(complex(0, phi_dash/2))]
                x_1 = cx_2_to_real_4(z_1)
                X_0 = two_balls(x_0)
                X_1 = two_balls(x_1)
                pts += Dot(X_0, color = BLUE)
                pts += Dot(X_1, color = RED)
        self.add(pts)

class StereogOrbitColours(ThreeDScene):
    def construct(self):
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        thetas = [np.pi/2 -np.pi/3, np.pi/2 , np.pi/2 + np.pi/3]
        phis = np.linspace(0, 2 * np.pi, 20)
        phi_dashes = np.linspace(0, 4 * np.pi, 20)
        Omegas = []
        for theta in thetas:
            for phi in phis:
                Omegas.append([theta, phi])
        Omegas = np.array(Omegas)

        pts = VGroup()
        for Omega in Omegas:
            for phi_dash in phi_dashes:
                x = angle_to_rep_4_vec(Omega)
                x_dash = rep_4_vec_orbit(x, phi_dash)
                X = stereog_proj(x_dash)

                theta = Omega[0]
                if theta < np.pi/2:
                    colour = RED
                if theta == np.pi/2:
                    colour = WHITE
                if theta > np.pi/2:
                    colour = BLUE
                pts += Dot(X, color = colour)
        self.add(pts)

class TwoBallsOrbitColours(ThreeDScene):
    def construct(self):
        phi = 90
        theta = 90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        thetas = [np.pi/2 -np.pi/3, np.pi/2 , np.pi/2 + np.pi/3]
        phis = np.linspace(0, 2 * np.pi, 20)
        phi_dashes = np.linspace(0, 4 * np.pi, 20)
        Omegas = []
        for theta in thetas:
            for phi in phis:
                Omegas.append([theta, phi])
        Omegas = np.array(Omegas)

        pts = VGroup()
        for Omega in Omegas:
            for phi_dash in phi_dashes:
                x = angle_to_rep_4_vec(Omega)
                x_dash = rep_4_vec_orbit(x, phi_dash)
                X = two_balls(x_dash)

                theta = Omega[0]
                if theta < np.pi/2:
                    colour = RED
                if theta == np.pi/2:
                    colour = WHITE
                if theta > np.pi/2:
                    colour = BLUE
                pts += Dot(X, color = colour)
        self.add(pts)
        
class Connection(ThreeDScene):
    def construct(self):
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        circle = Circle(radius=1.0).from_three_points(LEFT, RIGHT, UP, color=ORANGE, fill_opacity=0.5)
        normals = VGroup()

        for r in [1/4, 1/2, 3/4]:
            for theta in np.linspace(0, 2*np.pi, num=12):
                base = np.array([r * np.cos(theta), r * np.sin(theta), 0])
                end = base + np.array([0, 0, -0.5])
                normal = Arrow(start = base, end = end, buff=0, stroke_width=2,
                max_tip_length_to_length_ratio=0.1)
                normals += normal

        self.add(normals, circle)


# Moving scenes

class StereogOrbit3DIllusion(ThreeDScene):
    def construct(self):
        phi = 75
        theta = 90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        thetas = [-np.pi/3]
        phis = np.linspace(0, 2 * np.pi, 20)
        phi_dashes = np.linspace(0, 4 * np.pi, 20)
        Omegas = []
        for theta in thetas:
            for phi in phis:
                Omegas.append([theta, phi])
        Omegas = np.array(Omegas)
        pts = VGroup()
        for Omega in Omegas:
            for phi_dash in phi_dashes:
                x = angle_to_rep_4_vec(Omega)
                x_dash = rep_4_vec_orbit(x, phi_dash)
                X = stereog_proj(x_dash)
                pts += Dot(X)
        self.add(pts)
        self.begin_3dillusion_camera_rotation(rate=2)
        self.wait(PI/2)
        self.stop_3dillusion_camera_rotation()

class TwoBallsOrbit3DIllusion(ThreeDScene):
    def construct(self):
        phi = 75
        theta = 90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        thetas = [-np.pi/3]
        phis = np.linspace(0, 2 * np.pi, 20)
        phi_dashes = np.linspace(0, 4 * np.pi, 20)
        Omegas = []
        for theta in thetas:
            for phi in phis:
                Omegas.append(np.array([theta, phi]))
        Omegas = np.array(Omegas)

        rotated_Omegas = []
        rot_vec = [0]*3 # np.pi/2 * np.array([1,0,0])
        rotation = ax_angle_to_mat(rot_vec)
        for Omega in Omegas:
            n = angle_to_3_vec(Omega)
            n_rot = np.dot(rotation, n)
            rotated_Omega = vec_to_angle(n_rot)
            rotated_Omegas.append(rotated_Omega)

        pts = VGroup()
        for Omega in rotated_Omegas:
            for phi_dash in phi_dashes:
                x = angle_to_rep_4_vec(Omega)
                x_dash = rep_4_vec_orbit(x, phi_dash)
                X = two_balls(x_dash)
                pts += Dot(X)
        self.add(pts)
        self.begin_3dillusion_camera_rotation(rate=2)
        self.wait(PI/2)
        self.stop_3dillusion_camera_rotation()

class StereogOrbitColours3DIllusion(ThreeDScene):
    def construct(self):
        phi = 15
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        thetas = [np.pi/2-np.pi/3, np.pi/2 + 0, np.pi/2 + np.pi/3]
        phis = np.linspace(0, 2 * np.pi, 20)
        phi_dashes = np.linspace(0, 4 * np.pi, 20)
        Omegas = []
        for theta in thetas:
            for phi in phis:
                Omegas.append([theta, phi])
        Omegas = np.array(Omegas)
        pts = VGroup()
        for Omega in Omegas:
            for phi_dash in phi_dashes:
                x = angle_to_rep_4_vec(Omega)
                x_dash = rep_4_vec_orbit(x, phi_dash)
                X = stereog_proj(x_dash)

                theta = Omega[0]
                if theta < np.pi/2:
                    colour = RED
                if theta == np.pi/2:
                    colour = WHITE
                if theta > np.pi/2:
                    colour = BLUE
                pts += Dot(X, color = colour)
        self.add(pts)
        self.begin_3dillusion_camera_rotation(rate=2)
        self.wait(PI/2)
        self.stop_3dillusion_camera_rotation()

class TwoBallsColours3DIllusion(ThreeDScene):
    def construct(self):
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        thetas = [np.pi/2-np.pi/3, np.pi/2 + 0, np.pi/2 + np.pi/3]
        phis = np.linspace(0, 2 * np.pi, 20)
        phi_dashes = np.linspace(0, 4 * np.pi, 20)
        Omegas = []
        for theta in thetas:
            for phi in phis:
                Omegas.append([theta, phi])
        Omegas = np.array(Omegas)
        pts = VGroup()
        for Omega in Omegas:
            for phi_dash in phi_dashes:
                x = angle_to_rep_4_vec(Omega)
                x_dash = rep_4_vec_orbit(x, phi_dash)
                X = two_balls(x_dash)

                theta = Omega[0]
                if theta < np.pi/2:
                    colour = RED
                if theta == np.pi/2:
                    colour = WHITE
                if theta > np.pi/2:
                    colour = BLUE
                pts += Dot(X, color = colour)
        self.add(pts)
        self.begin_3dillusion_camera_rotation(rate=2)
        self.wait(PI/2)
        self.stop_3dillusion_camera_rotation()

class StereogOrbitColoursTransform(ThreeDScene):
    def construct(self):
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        thetas = [np.pi/2-np.pi/3, np.pi/2 + 0, np.pi/2 + np.pi/3]
        phis = np.linspace(0, 2 * np.pi, 20)
        phi_dashes = np.linspace(0, 4 * np.pi, 100)
        Omegas = []
        for theta in thetas:
            for phi in phis:
                Omegas.append([theta, phi])
        Omegas = np.array(Omegas)

        rotated_Omegas = []
        rot_vec = np.pi/2 * np.array([0,0,0])
        rotation = ax_angle_to_mat(rot_vec)
        for Omega in Omegas:
            n = angle_to_3_vec(Omega)
            n_rot = np.dot(rotation, n)
            rotated_Omega = vec_to_angle(n_rot)
            rotated_Omegas.append(rotated_Omega)

        pts = VGroup()
        pts_red = VGroup()
        pts_white = VGroup()
        pts_blue = VGroup()
        for Omega in rotated_Omegas:
            for phi_dash in phi_dashes:
                x = angle_to_rep_4_vec(Omega)
                x_dash = rep_4_vec_orbit(x, phi_dash)
                X = stereog_proj(x_dash)

                theta = Omega[0]
                if theta < np.pi/2:
                    colour = RED
                    pts += Dot(X, color = colour)
                    pts_red += Dot(X, color = colour)
                if theta == np.pi/2:
                    colour = WHITE
                    pts_white += Dot(X, color = colour)
                if theta > np.pi/2:
                    colour = BLUE
                    pts_blue += Dot(X, color = colour)
        self.add(pts)
        self.begin_ambient_camera_rotation(rate=np.pi/6)
        self.wait()
        self.play(Transform(pts, pts_white))
        self.wait()
        self.play(Transform(pts, pts_blue))
        self.wait()
        self.play(Transform(pts, pts_red))

class TwoBallsOrbitColoursTransform(ThreeDScene):
    def construct(self):
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        thetas = [np.pi/2-np.pi/3, np.pi/2 + 0, np.pi/2 + np.pi/3]
        phis = np.linspace(0, 2 * np.pi, 20)
        phi_dashes = np.linspace(0, 4 * np.pi, 20)
        Omegas = []
        for theta in thetas:
            for phi in phis:
                Omegas.append([theta, phi])
        Omegas = np.array(Omegas)

        rotated_Omegas = []
        rot_vec = np.pi/2 * np.array([0,0,0])
        rotation = ax_angle_to_mat(rot_vec)
        for Omega in Omegas:
            n = angle_to_3_vec(Omega)
            n_rot = np.dot(rotation, n)
            rotated_Omega = vec_to_angle(n_rot)
            rotated_Omegas.append(rotated_Omega)

        pts = VGroup()
        pts_red = VGroup()
        pts_white = VGroup()
        pts_blue = VGroup()
        for Omega in rotated_Omegas:
            for phi_dash in phi_dashes:
                x = angle_to_rep_4_vec(Omega)
                x_dash = rep_4_vec_orbit(x, phi_dash)
                X = two_balls(x_dash)

                theta = Omega[0]
                if theta < np.pi/2:
                    colour = RED
                    pts += Dot(X, color = colour)
                    pts_red += Dot(X, color = colour)
                if theta == np.pi/2:
                    colour = WHITE
                    pts_white += Dot(X, color = colour)
                if theta > np.pi/2:
                    colour = BLUE
                    pts_blue += Dot(X, color = colour)
        self.add(pts)
        self.begin_ambient_camera_rotation(rate=np.pi/6)
        self.wait()
        self.play(Transform(pts, pts_white))
        self.wait()
        self.play(Transform(pts, pts_blue))
        self.wait()
        self.play(Transform(pts, pts_red))

class SO3ProjOrbitColoursTransform(ThreeDScene):
    def construct(self):
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        thetas = [np.pi/2-np.pi/3, np.pi/2 + 0, np.pi/2 + np.pi/3]
        phis = np.linspace(0, 2 * np.pi, 20)
        phi_dashes = np.linspace(0, 4 * np.pi, 100)
        Omegas = []
        for theta in thetas:
            for phi in phis:
                Omegas.append([theta, phi])
        Omegas = np.array(Omegas)

        rotated_Omegas = []
        rot_vec = np.pi/2 * np.array([0,0,0])
        rotation = ax_angle_to_mat(rot_vec)
        for Omega in Omegas:
            n = angle_to_3_vec(Omega)
            n_rot = np.dot(rotation, n)
            rotated_Omega = vec_to_angle(n_rot)
            rotated_Omegas.append(rotated_Omega)

        pts = VGroup()
        pts_red = VGroup()
        pts_white = VGroup()
        pts_blue = VGroup()
        for Omega in rotated_Omegas:
            for phi_dash in phi_dashes:
                x = angle_to_rep_4_vec(Omega)
                x_dash = rep_4_vec_orbit(x, phi_dash)
                X = SO3_proj(x_dash)

                theta = Omega[0]
                if theta < np.pi/2:
                    colour = RED
                    pts += Dot(X, color = colour)
                    pts_red += Dot(X, color = colour)
                if theta == np.pi/2:
                    colour = WHITE
                    pts_white += Dot(X, color = colour)
                if theta > np.pi/2:
                    colour = BLUE
                    pts_blue += Dot(X, color = colour)
        self.add(pts)
        self.begin_ambient_camera_rotation(rate=np.pi/6)
        self.wait()
        self.play(Transform(pts, pts_white))
        self.wait()
        self.play(Transform(pts, pts_blue))
        self.wait()
        self.play(Transform(pts, pts_red))

class HopfSpiralCreation(ThreeDScene):
    def construct(self):
        phi = -30
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        spiral = Surface(
            spiral_param,
            resolution=(100, 32),
            u_range=[0, 2/3*np.pi],
            v_range=[0, 4*np.pi],
        )
        spiral.set_fill_by_checkerboard(ORANGE, BLUE)

        self.play(Create(spiral), run_time = 4)
        self.play(FadeOut(spiral))

class FundamentalVectorField(ThreeDScene):
    def construct(self):
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)

        cloud = Mobject1D()
        circle = Circle(radius=1.0, color=ORANGE, fill_opacity=0.5)
        top_orbits = VGroup()
        bottom_orbits = VGroup()

        for r in [1/4, 1/2, 3/4]:
            if r < 1/2:
                colour = RED
            elif r == 1/2:
                colour = GREEN
            elif r > 1/2:
                colour = BLUE
            for theta in np.linspace(0, 2*np.pi, num=12):
                position = np.array([r * np.cos(theta), r * np.sin(theta), 0])
                bottom_orbit = ParametricFunction(
                    lambda phi_dash: fund_flow_stereog(position, phi_dash),
                    t_range = np.array([0, 2 * np.pi]),
                    color = colour
                )
                top_orbit = ParametricFunction(
                    lambda phi_dash: fund_flow_stereog(position, phi_dash),
                    t_range = np.array([2 * np.pi, 4 * np.pi]),
                    color = colour
                )
                bottom_orbits += bottom_orbit
                top_orbits += top_orbit
        cloud.add_points(
            [
                [r * np.cos(theta), r * np.sin(theta), 0]
                for r in [1/4, 1/2, 3/4]
                for theta in np.linspace(0, 2*np.pi, num=12)
            ]
        )

        self.add(bottom_orbits, circle, top_orbits, cloud)
        n = 24
        for i in range(n):
            self.play(
                ApplyPointwiseFunction(
                    lambda x: fund_flow_stereog(x, 4*np.pi/n),
                    cloud,
                    run_time=1,
                    lag_ratio=2
                )
            )

class RightEquivariance(ThreeDScene):
    def construct(self):
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(1.4)

        circle = Circle(radius=1.0).from_three_points(LEFT, RIGHT, UP, color=ORANGE, fill_opacity=0.5)
        circle_illusion = Circle(radius=1.0).from_three_points(LEFT, RIGHT, UP, color=ORANGE, fill_opacity=0.5)
        normals = VGroup()

        for r in [3/4, 1/2, 1/4]:
            for theta in np.linspace(0, 2*np.pi, num=12):
                base = np.array([r * np.cos(theta), r * np.sin(theta), 0])
                end = base + np.array([0, 0, -0.5])
                normal = Arrow(start = base, end = end, buff=0, stroke_width=2,
                max_tip_length_to_length_ratio=0.1)
                normals += normal
        normal = Arrow(start = base, end = end, buff=0, stroke_width=2,
                max_tip_length_to_length_ratio=0.1, color = RED)
        horizontal = Square(side_length=0.5, color = RED, fill_opacity = 0.5, stroke_width = 1).shift(base)

        spanners_0 = VGroup()
        spanners_0 += Arrow(start = base, end = base + RIGHT*0.5, buff=0, stroke_width=2,
                max_tip_length_to_length_ratio=0.1, color = RED)
        spanners_0 += Arrow(start = base, end = base + UP*0.5, buff=0, stroke_width=2,
                max_tip_length_to_length_ratio=0.1, color = RED)

        p_g = fund_flow_stereog(base, np.pi)
        spanners_1 = VGroup()
        v_1 = np.array([0.4, 0, -0.2])
        v_2 = np.array([0, -0.4, -0.2])
        spanners_1 += Arrow(start = p_g, end = p_g + v_1, buff=0, stroke_width=2,
                max_tip_length_to_length_ratio=0.1, color = RED)
        spanners_1 += Arrow(start = p_g, end = p_g + v_2, buff=0, stroke_width=2,
                max_tip_length_to_length_ratio=0.1, color = RED)
        v_3 = np.cross(v_1, v_2) * 3
        normal_1 = Arrow(start = p_g, end = p_g + v_3, buff=0, stroke_width=2,
                max_tip_length_to_length_ratio=0.1, color = RED)

        horizontal_1 = Surface(
            lambda u,v: p_g + u * v_1 + v * v_2,
            u_range = [-.5, .5],
            v_range = [-.5, .5],
            resolution = (1,1),
        ).set_fill(RED)
        
        bottom_orbit = ParametricFunction(
                    lambda phi_dash: fund_flow_stereog(base, phi_dash),
                    t_range = np.array([0, 2 * np.pi]),
                    color = GREEN
                )
        top_orbit = ParametricFunction(
                    lambda phi_dash: fund_flow_stereog(base, phi_dash),
                    t_range = np.array([2 * np.pi, 4 * np.pi]),
                    color = GREEN
                )
        self.begin_ambient_camera_rotation(rate=np.pi/12)
        self.play(FadeIn(normals, normal, circle))
        self.play(FadeOut(normals))
        self.play(FadeIn(bottom_orbit, circle_illusion, top_orbit), FadeOut(circle))
        self.play(FadeIn(horizontal), FadeOut(normal))
        self.play(FadeIn(spanners_0), FadeOut(horizontal))
        self.play(Transform(spanners_0, spanners_1))
        self.play(FadeOut(spanners_0), FadeIn(horizontal_1))
        self.play(FadeOut(horizontal_1), FadeIn(normal_1))
        self.wait()
        self.play(FadeOut(circle_illusion, normal_1, bottom_orbit, top_orbit))

class Projection(ThreeDScene):
    def construct(self):
        phi = 75
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(5)

        self.begin_ambient_camera_rotation(rate=np.pi/12)

        n = np.array([1/3, 1/3, 1/3])
        xi = np.array([0, 1/4, -1/2])
        v = np.array([-1/3, 2/3, -1/3])

        normals = VGroup()

        for r in [1/4]:
            for theta in [0]:
                base = np.array([r * np.cos(theta), r * np.sin(theta), 0])
                end = base + np.array([0, 0, -0.5])
                normal = Arrow(start = base, end = end, buff=0, stroke_width=2,
                max_tip_length_to_length_ratio=0.1)
                normals += normal
        normal = Arrow(start = base, end = end, buff=0, stroke_width=2,
                max_tip_length_to_length_ratio=0.1, color = RED)
        bottom_orbit = ParametricFunction(
                    lambda phi_dash: fund_flow_stereog(base, phi_dash),
                    t_range = np.array([0, 2 * np.pi]),
                    color = GREEN
                )
        top_orbit = ParametricFunction(
                    lambda phi_dash: fund_flow_stereog(base, phi_dash),
                    t_range = np.array([2 * np.pi, 4 * np.pi]),
                    color = GREEN
                )
        xi_arrow = Arrow(start = base, end = base + xi, buff=0, stroke_width=2,
                max_tip_length_to_length_ratio=0.1, color = GREEN)

        horizontal = Square(side_length=0.5, color = RED, fill_opacity = 0.5, stroke_width = 1).shift(base)

        horizontal_2 = Square(side_length=0.5, color = RED, fill_opacity = 0.5, stroke_width = 1).shift(base)

        level = Surface(
            lambda u,v: base + u * RIGHT + v * UP - 1/3 * np.array([0,0,1]),
            u_range = [.5, -.5],
            v_range = [.5, -.5],
            resolution = (1,1),
            fill_opacity=0.5
        ).set_fill(RED)

        v_arrow = Arrow(start = base, end = base + v, buff=0, stroke_width=2,
                max_tip_length_to_length_ratio=0.1, color = PURPLE)
        v_1 = np.array([0, 1/6, -1/3])
        v_arrow_1 = Arrow(start = base, end = base + v_1, buff=0, stroke_width=2,
                max_tip_length_to_length_ratio=0.1, color = PURPLE)

        self.play(FadeIn(normal, bottom_orbit, top_orbit, v_arrow))
        self.play(FadeIn(xi_arrow))
        self.play(FadeOut(bottom_orbit, top_orbit))
        self.play(FadeIn(horizontal, horizontal_2))
        self.play(Transform(horizontal_2, level))
        self.play(Transform(v_arrow, v_arrow_1))
        self.wait(2)
        self.play(FadeOut(horizontal, horizontal_2, normal, v_arrow, xi_arrow))

