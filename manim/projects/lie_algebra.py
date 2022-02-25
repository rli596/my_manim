###
# lie_algebra.py
#
# Visualising left invariant vector fields
###

'''
Dependencies
'''

from manim import *

'''
Functions
'''

# Parametrise surface
def param_sphere(u, v):
    R = np.pi
    x = R * np.sin(v) * np.cos(u)
    y = R * np.sin(v) * np.sin(u)
    z = R * np.cos(v)
    return np.array([x,y,z])

# Linear algebra calculations
def skew_to_vec(skew):
    '''
    Parameters
    ----------
    skew: np.array
        3x3 skew symmetric matrix

    Returns
    -------
    vec: np.array
        3D vector
    '''
    x = skew 


class SO3(ThreeDScene):
    def construct(self):
        resolution_fa = 10
        self.set_camera_orientation(phi = 60 * DEGREES)
        
        sphere = Surface(
            param_sphere,
            resolution=(resolution_fa, resolution_fa),
            v_range=[0, np.pi],
            u_range=[-np.pi, np.pi]
        )
        vec = Arrow3D(start = np.array([0,0,0]), 
        end = np.array([0,0,1]))
        sphere.set_fill_by_checkerboard(BLUE, BLUE, opacity = 0.5)

        self.add(sphere, vec)