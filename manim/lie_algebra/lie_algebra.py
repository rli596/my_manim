###
# lie_algebra.py
#
# Visualising left invariant vector fields
###

'''
Dependencies
'''

from manim import *
from utils import *
from functools import partial

'''
Scenes
'''

class SO3(ThreeDScene):
    def construct(self):
        resolution_fa = 10
        self.set_camera_orientation(phi = 60 * DEGREES, theta = 30*DEGREES)
        
        sphere = Surface(
            param_sphere,
            resolution=(resolution_fa, resolution_fa),
            v_range=[0, np.pi],
            u_range=[-np.pi, np.pi]
        )
        start = np.array([0,0,0])
        end = np.array([0,0,1])
        """ 
        vec = Arrow3D(start = start, 
        end = end)
         """
        sphere.set_fill_by_checkerboard(BLUE, BLUE, opacity = 0.5)

        vec_2 = ParametricFunction(partial(param_line, start, end), [0,1]).set_color(
            WHITE
        )

        vec_3 = Vector(end - start).shift(start)

        self.add(sphere, vec_3)

class SO3_push_one(ThreeDScene):
    def construct(self):
        resolution_fa = 10
        self.set_camera_orientation(phi = 75 * DEGREES, theta = 30 * DEGREES)
        arrow_length = 1
        
        sphere = Surface(
            param_sphere,
            resolution=(resolution_fa, resolution_fa),
            v_range=[0, np.pi],
            u_range=[-np.pi, np.pi]
        )
        
        origin = np.array([0,0,0])
        X_z = np.array([0,0,1])

        vec = Arrow3D(start = origin, 
        end = origin + X_z)

        p = np.array([np.pi/3,0,np.pi/3])
        X_p = pushforward_by_ax_angle(X_z, p)
        vec2 = Arrow3D(start = p,
        end = p + X_p)

        sphere.set_fill_by_checkerboard(BLUE, BLUE, opacity = 0.5)

        self.add(sphere, vec, vec2)

class SO3_push_set_xz_plane(ThreeDScene):
    def construct(self):
        resolution_fa = 10
        phi = 90
        theta = 180
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        arrow_length = 0.5
        
        sphere = Surface(
            param_sphere,
            resolution=(resolution_fa, resolution_fa),
            v_range=[0, np.pi],
            u_range=[-np.pi, np.pi]
        )
        sphere.set_fill_by_checkerboard(BLUE, BLUE, opacity = 0.5)
        
        ps = []
        z_interval = np.pi/3
        z_num_vals = 25
        x_interval = np.pi/3
        x_num_vals = 11
        for z in np.linspace(-np.pi, np.pi, z_num_vals):
            for x in np.linspace(0, np.sqrt(np.pi**2 - z**2), x_num_vals):
                ps.append([x, 0, z])
        ps = np.array(ps)

        config.output_file = f'v_field_{phi}_{theta}_{z_num_vals}_{x_num_vals}'

        X_0 = [0,0,1]

        pairs = pushforward_point_vec_pairs(X_0, ps)
        vecs = {}

        v_field = VGroup()
        for pair in pairs:
            vecs[tuple(pair[0])] = Vector(
                pair[1]*arrow_length,
                color = interpolate_color(PURPLE, YELLOW, (pair[0][2]+np.pi)/(2*np.pi))
            ).shift(pair[0])
            v_field += vecs[tuple(pair[0])]

        self.add(sphere, v_field)

class SO3_push_set_whole_sphere(ThreeDScene):
    def construct(self):
        resolution_fa = 10
        self.set_camera_orientation(phi = 75 * DEGREES, theta = 30 * DEGREES)
        arrow_length = 0.5
        
        sphere = Surface(
            param_sphere,
            resolution=(resolution_fa, resolution_fa),
            v_range=[0, np.pi],
            u_range=[-np.pi, np.pi]
        )
        sphere.set_fill_by_checkerboard(BLUE, BLUE, opacity = 0.5)
        self.add(sphere)
        
        ps = []
        z_interval = np.pi/3
        z_num_vals = 7
        x_interval = np.pi/3
        x_num_vals = 3
        theta_num_vals = 7 # In practice the number of distinct angles will be this - 1
        for z in np.linspace(-np.pi, np.pi, z_num_vals):
            for x in np.linspace(0, np.sqrt(np.pi**2 - z**2), x_num_vals):
                for theta in np.linspace(-np.pi, np.pi, theta_num_vals):
                    ps.append([x*np.cos(theta), x*np.sin(theta), z])
        ps = np.array(ps)

        X_0 = [0,0,1]

        pairs = pushforward_point_vec_pairs(X_0, ps)
        vecs = {}

        v_field = VGroup()
        for pair in pairs:
            vecs[tuple(pair[0])] = Vector(
                pair[1]*arrow_length,
                color = interpolate_color(PURPLE, YELLOW, (pair[0][2]+np.pi)/(2*np.pi))
            ).shift(pair[0])
            v_field += vecs[tuple(pair[0])]

        self.add(sphere, v_field)

class SO3_push_one_anim(ThreeDScene):
    def construct(self):
        resolution_fa = 10
        self.set_camera_orientation(phi = 75 * DEGREES, theta = 30 * DEGREES)
        arrow_length = 1
        
        sphere = Surface(
            param_sphere,
            resolution=(resolution_fa, resolution_fa),
            v_range=[0, np.pi],
            u_range=[-np.pi, np.pi]
        )
        
        origin = np.array([0,0,0])
        X_z = np.array([0,0,1])

        vec = Arrow3D(start = origin, 
        end = origin + X_z)

        p = np.array([np.pi/3,0,np.pi/3])
        X_p = pushforward_by_ax_angle(X_z, p)
        vec2 = Arrow3D(start = p,
        end = p + X_p)

        sphere.set_fill_by_checkerboard(BLUE, BLUE, opacity = 0.5)

        self.add(sphere, vec)
        self.play(Transform(vec, vec2))

class SO3_act_with_z_rot(ThreeDScene):
    def construct(self):
        resolution_fa = 12
        self.set_camera_orientation(phi = 90 * DEGREES, theta = 30 * DEGREES)
        arrow_length = 0.5
        
        sphere = Surface(
            param_sphere,
            resolution=(resolution_fa, resolution_fa),
            v_range=[0, np.pi],
            u_range=[-np.pi, np.pi]
        )
        sphere.set_fill_by_checkerboard(BLUE, BLUE, opacity = 0.5)
        self.add(sphere)
        
        ps = []
        z_num_vals = 25
        x_num_vals = 11
        theta_num_vals = 12
        # Add vectors to vgroup
        for z in np.linspace(-np.pi*(1 - 1/z_num_vals), \
            np.pi*(1 - 1/z_num_vals), z_num_vals):
            ps.append([0, 0, z])
        for z in np.linspace(-np.pi*(1 - 1/z_num_vals), \
            np.pi*(1 - 1/z_num_vals), z_num_vals):
            for x in np.linspace(np.sqrt(np.pi**2 - z**2)/x_num_vals, \
                np.sqrt(np.pi**2 - z**2)*(1 - 1/x_num_vals), x_num_vals - 1):
                for theta in np.linspace(0, 2*np.pi*(1-1/theta_num_vals), \
                    theta_num_vals):
                    ps.append([x*np.cos(theta), x*np.sin(theta), z])
        ps = np.array(ps)

        X_0 = np.array([0,0,1])

        pairs = pushforward_point_vec_pairs(X_0, ps)
        vecs = {}
        v_field = VGroup()
        for pair in pairs:
            vecs[tuple(pair[0])] = Vector(
                pair[1]*arrow_length,
                color = interpolate_color(PURPLE, YELLOW, (pair[0][2]+np.pi)/(2*np.pi))
            ).shift(pair[0])
            v_field += vecs[tuple(pair[0])]

        R_vec = np.array([0,0,np.pi/3])

        pushed_pairs = pushforward_vector_field(pairs, R_vec)
        pushed_vecs = {}
        pushed_v_field = VGroup()
        for pushed_pair in pushed_pairs:
            pushed_vecs[tuple(pushed_pair[0])] = Vector(
                pushed_pair[1]*arrow_length,
                color = interpolate_color(PURPLE, YELLOW, (pushed_pair[0][2]+np.pi)/(2*np.pi))
            ).shift(pushed_pair[0])
            pushed_v_field += pushed_vecs[tuple(pushed_pair[0])]

        self.add(sphere, v_field)
        self.play(Transform(v_field, pushed_v_field))
        
        i = 0
        for i in range(5):
            pushed_pairs = pushforward_vector_field(pushed_pairs, R_vec)
            pushed_vecs = {}
            pushed_v_field = VGroup()
            for pushed_pair in pushed_pairs:
                pushed_vecs[tuple(pushed_pair[0])] = Vector(
                    pushed_pair[1]*arrow_length,
                    color = interpolate_color(PURPLE, YELLOW, (pushed_pair[0][2]+np.pi)/(2*np.pi))
                ).shift(pushed_pair[0])
                pushed_v_field += pushed_vecs[tuple(pushed_pair[0])]
            self.play(Transform(v_field, pushed_v_field))

class SO3_plane_act_with_z_rot(ThreeDScene):
    def construct(self):
        resolution_fa = 12
        self.set_camera_orientation(phi = 60 * DEGREES, theta = 0 * DEGREES)
        arrow_length = 0.5
        
        sphere = Surface(
            param_sphere,
            resolution=(resolution_fa, resolution_fa),
            v_range=[0, np.pi],
            u_range=[-np.pi, np.pi]
        )
        sphere.set_fill_by_checkerboard(BLUE, BLUE, opacity = 0.5)
        self.add(sphere)
        
        ps = []
        z_num_vals = 6
        x_num_vals = 3
        theta_num_vals = 1 # In practice the number of distinct angles will be this - 1
        # Add vectors to vgroup
        for z in np.linspace(-np.pi*(1 - 2/z_num_vals), np.pi, z_num_vals):
            ps.append([0, 0, z])
        for z in np.linspace(-np.pi*(1 - 2/z_num_vals), np.pi, z_num_vals):
            for x in np.linspace(np.pi/3, np.sqrt(np.pi**2 - z**2), x_num_vals):
                for theta in np.linspace(0, 2*np.pi*(1-1/theta_num_vals), \
                    theta_num_vals):
                    ps.append([x*np.cos(theta), x*np.sin(theta), z])
        ps = np.array(ps)

        X_0 = np.array([0,0,1])

        pairs = pushforward_point_vec_pairs(X_0, ps)
        vecs = {}
        v_field = VGroup()
        for pair in pairs:
            vecs[tuple(pair[0])] = Vector(
                pair[1]*arrow_length,
                color = interpolate_color(PURPLE, YELLOW, (pair[0][2]+np.pi)/(2*np.pi))
            ).shift(pair[0])
            v_field += vecs[tuple(pair[0])]

        R_vec = np.array([0,0,np.pi/3])

        pushed_pairs = pushforward_vector_field(pairs, R_vec)
        pushed_vecs = {}
        pushed_v_field = VGroup()
        for pushed_pair in pushed_pairs:
            pushed_vecs[tuple(pushed_pair[0])] = Vector(
                pushed_pair[1]*arrow_length,
                color = interpolate_color(PURPLE, YELLOW, (pushed_pair[0][2]+np.pi)/(2*np.pi))
            ).shift(pushed_pair[0])
            pushed_v_field += pushed_vecs[tuple(pushed_pair[0])]

        self.add(sphere, v_field)
        self.play(Transform(v_field, pushed_v_field))
        
        i = 0
        for i in range(5):
            pushed_pairs = pushforward_vector_field(pushed_pairs, R_vec)
            pushed_vecs = {}
            pushed_v_field = VGroup()
            for pushed_pair in pushed_pairs:
                pushed_vecs[tuple(pushed_pair[0])] = Vector(
                    pushed_pair[1]*arrow_length,
                    color = interpolate_color(PURPLE, YELLOW, (pushed_pair[0][2]+np.pi)/(2*np.pi))
                ).shift(pushed_pair[0])
                pushed_v_field += pushed_vecs[tuple(pushed_pair[0])]
            self.play(Transform(v_field, pushed_v_field))

class SO3_act_with_x_rot(ThreeDScene):
    def construct(self):
        resolution_fa = 12
        self.set_camera_orientation(phi = 90 * DEGREES, theta = 30 * DEGREES)
        arrow_length = 0.5
        
        sphere = Surface(
            param_sphere,
            resolution=(resolution_fa, resolution_fa),
            v_range=[0, np.pi],
            u_range=[-np.pi, np.pi]
        )
        sphere.set_fill_by_checkerboard(BLUE, BLUE, opacity = 0.5)
        self.add(sphere)
        
        ps = []
        z_num_vals = 6
        x_num_vals = 3
        theta_num_vals = 6 # In practice the number of distinct angles will be this - 1
        # Add vectors to vgroup
        for z in np.linspace(-np.pi*(1 - 1/z_num_vals), \
            np.pi*(1 - 1/z_num_vals), z_num_vals):
            ps.append([0, 0, z])
        for z in np.linspace(-np.pi*(1 - 1/z_num_vals), \
            np.pi*(1 - 1/z_num_vals), z_num_vals):
            for x in np.linspace(np.sqrt(np.pi**2 - z**2)/x_num_vals, \
                np.sqrt(np.pi**2 - z**2)*(1 - 1/x_num_vals), x_num_vals - 1):
                for theta in np.linspace(0, 2*np.pi*(1-1/theta_num_vals), \
                    theta_num_vals):
                    ps.append([x*np.cos(theta), x*np.sin(theta), z])
        ps = np.array(ps)

        X_0 = np.array([0,0,1])

        pairs = pushforward_point_vec_pairs(X_0, ps)
        vecs = {}
        v_field = VGroup()
        for pair in pairs:
            vecs[tuple(pair[0])] = Vector(
                pair[1]*arrow_length,
                color = interpolate_color(PURPLE, YELLOW, (pair[0][2]+np.pi)/(2*np.pi))
            ).shift(pair[0])
            v_field += vecs[tuple(pair[0])]

        R_vec = np.array([np.pi/3,0,0])

        pushed_pairs = pushforward_vector_field(pairs, R_vec)
        pushed_vecs = {}
        pushed_v_field = VGroup()
        for pushed_pair in pushed_pairs:
            pushed_vecs[tuple(pushed_pair[0])] = Vector(
                pushed_pair[1]*arrow_length,
                color = interpolate_color(PURPLE, YELLOW, (pushed_pair[0][2]+np.pi)/(2*np.pi))
            ).shift(pushed_pair[0])
            pushed_v_field += pushed_vecs[tuple(pushed_pair[0])]

        self.add(sphere, v_field)
        self.play(Transform(v_field, pushed_v_field))
        
        i = 0
        for i in range(5):
            pushed_pairs = pushforward_vector_field(pushed_pairs, R_vec)
            pushed_vecs = {}
            pushed_v_field = VGroup()
            for pushed_pair in pushed_pairs:
                pushed_vecs[tuple(pushed_pair[0])] = Vector(
                    pushed_pair[1]*arrow_length,
                    color = interpolate_color(PURPLE, YELLOW, (pushed_pair[0][2]+np.pi)/(2*np.pi))
                ).shift(pushed_pair[0])
                pushed_v_field += pushed_vecs[tuple(pushed_pair[0])]
            self.play(Transform(v_field, pushed_v_field))

class SO3_act_with_rot(ThreeDScene):
    def construct(self):
        resolution_fa = 12
        self.set_camera_orientation(phi = 90 * DEGREES, theta = 30 * DEGREES)
        arrow_length = 0.5
        
        sphere = Surface(
            param_sphere,
            resolution=(resolution_fa, resolution_fa),
            v_range=[0, np.pi],
            u_range=[-np.pi, np.pi]
        )
        sphere.set_fill_by_checkerboard(BLUE, BLUE, opacity = 0.5)
        self.add(sphere)
        
        ps = []
        z_num_vals = 6
        x_num_vals = 3
        theta_num_vals = 6 # In practice the number of distinct angles will be this - 1
        # Add vectors to vgroup
        for z in np.linspace(-np.pi*(1 - 1/z_num_vals), \
            np.pi*(1 - 1/z_num_vals), z_num_vals):
            ps.append([0, 0, z])
        for z in np.linspace(-np.pi*(1 - 1/z_num_vals), \
            np.pi*(1 - 1/z_num_vals), z_num_vals):
            for x in np.linspace(np.sqrt(np.pi**2 - z**2)/x_num_vals, \
                np.sqrt(np.pi**2 - z**2)*(1 - 1/x_num_vals), x_num_vals - 1):
                for theta in np.linspace(0, 2*np.pi*(1-1/theta_num_vals), \
                    theta_num_vals):
                    ps.append([x*np.cos(theta), x*np.sin(theta), z])
        ps = np.array(ps)

        X_0 = np.array([0,0,1])

        pairs = pushforward_point_vec_pairs(X_0, ps)
        vecs = {}
        v_field = VGroup()
        for pair in pairs:
            vecs[tuple(pair[0])] = Vector(
                pair[1]*arrow_length,
                color = interpolate_color(PURPLE, YELLOW, (pair[0][2]+np.pi)/(2*np.pi))
            ).shift(pair[0])
            v_field += vecs[tuple(pair[0])]

        R_vec = np.pi/3 * np.array([np.sqrt(1/3),np.sqrt(1/3),np.sqrt(1/3)])

        pushed_pairs = pushforward_vector_field(pairs, R_vec)
        pushed_vecs = {}
        pushed_v_field = VGroup()
        for pushed_pair in pushed_pairs:
            pushed_vecs[tuple(pushed_pair[0])] = Vector(
                pushed_pair[1]*arrow_length,
                color = interpolate_color(PURPLE, YELLOW, (pushed_pair[0][2]+np.pi)/(2*np.pi))
            ).shift(pushed_pair[0])
            pushed_v_field += pushed_vecs[tuple(pushed_pair[0])]

        self.add(sphere, v_field)
        self.play(Transform(v_field, pushed_v_field))
        
        i = 0
        for i in range(5):
            pushed_pairs = pushforward_vector_field(pushed_pairs, R_vec)
            pushed_vecs = {}
            pushed_v_field = VGroup()
            for pushed_pair in pushed_pairs:
                pushed_vecs[tuple(pushed_pair[0])] = Vector(
                    pushed_pair[1]*arrow_length,
                    color = interpolate_color(PURPLE, YELLOW, (pushed_pair[0][2]+np.pi)/(2*np.pi))
                ).shift(pushed_pair[0])
                pushed_v_field += pushed_vecs[tuple(pushed_pair[0])]
            self.play(Transform(v_field, pushed_v_field))