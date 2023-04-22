'''
boys.py

Scenes for boys surface visualization
'''

'''
Dependencies
'''

from boys_utils import *
from manim import *

'''
Scenes
'''

class BoysSurface(ThreeDScene):
    def construct(self):
        # Camera
        phi = 0
        theta = -90
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(2)

        # Mobjects
        BK_boys = Surface(lambda u,v: BK_param(np.array([u * np.cos(v), u * np.sin(v), 0])),
                          u_range = [0,1],
                          v_range = [0, 2 * np.pi],
                          resolution = (24,24),
                          fill_opacity=0.7)
        
        # Animations
        self.add(BK_boys)
        self.begin_ambient_camera_rotation(about='phi', rate = np.pi/6)
        self.wait(3)
        self.stop_ambient_camera_rotation(about='phi')
        self.wait()
        self.begin_ambient_camera_rotation(about='theta', rate = np.pi/6)
        self.wait(12)
        self.stop_ambient_camera_rotation(about='theta')
        self.wait()

class BoysSurfaceStill(ThreeDScene):
    def construct(self):
        # Camera
        phi = 0
        theta = 0
        self.set_camera_orientation(phi = phi*DEGREES, theta = theta*DEGREES)
        self.camera.set_zoom(2)

        # Mobjects
        BK_boys = Surface(lambda u,v: BK_param(np.array([u * np.cos(v), u * np.sin(v), 0])),
                          u_range = [0,1],
                          v_range = [0, 2 * np.pi],
                          resolution = (64,24),
                          fill_opacity=0.01)
        
        # Animations
        self.add(BK_boys)