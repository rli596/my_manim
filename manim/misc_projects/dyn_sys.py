####
# dyn_sys.py
#
# Playing with dynamical systems
####

'''
Dependencies
'''
from manim import *

'''
Functions
'''

def V_1(pos):
    x = pos[0]
    y = pos[1]
    f = np.sin(x/2)
    g = np.cos(y/2)
    V = f * UP + g * LEFT
    return V

def renorm_flow(pos):
    LAMBDA = 1
    EPSILON = 1e-2
    mu_sq = pos[0]
    g = pos[1]
    V_1 = 2 * mu_sq + (3/(2 * np.pi**2)) * (LAMBDA**4/(LAMBDA**2 + mu_sq)) * g
    V_2 = EPSILON * g - (9 /(2 * np.pi**2)) * (LAMBDA**4/(LAMBDA**2 + mu_sq)**2) * g**2
    V = V_1 * UP + V_2 * RIGHT
    return V

def van_der_pol(pos):
    # Lienard form
    MU = 2
    x = 5 * pos[0]
    y = 5 * pos[1]
    f = MU * (x - 1/3 * x **3 - y)
    g = x/MU
    V = f * UP + g * RIGHT
    return V


'''
Config
'''


'''
Scenes
'''

class BasicUsage(Scene):
    def construct(self):
        func = lambda pos: ((pos[0] * UR + pos[1] * LEFT) - pos) / 3
        self.add(StreamLines(func))

class SpawningAndFlowingArea(Scene):
    def construct(self):
        func = lambda pos: np.sin(pos[0]) * UR + np.cos(pos[1]) * LEFT + pos / 5
        stream_lines = StreamLines(
            func, x_range=[-3, 3, 0.2], y_range=[-2, 2, 0.2], padding=1
        )

        spawning_area = Rectangle(width=6, height=4)
        #flowing_area = Rectangle(width=8, height=6)
        labels = [Tex("Spawning Area"), Tex("Flowing Area").shift(DOWN * 2.5)]
        for lbl in labels:
            lbl.add_background_rectangle(opacity=0.6, buff=0.05)

        self.add(stream_lines, spawning_area, 
        #flowing_area, 
        *labels)

class StreamLineCreation(Scene):
    def construct(self):
        stream_lines = StreamLines(
            renorm_flow,
            color=YELLOW,
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            stroke_width=3,
            virtual_time=1,  # use shorter lines
            max_anchors_per_line=5,  # better performance with fewer anchors
        )
        self.play(stream_lines.create())  # uses virtual_time as run_time
        self.wait()

class EndAnimation(Scene):
    def construct(self):
        stream_lines = StreamLines(
            renorm_flow, 
            x_range = [0, 2, 0.2],
            y_range = [0, 2, 0.2],
            stroke_width=3, max_anchors_per_line=5, virtual_time=1, color = BLUE
        )
        axes = Axes(
            x_range = [0,2,0.2],
            y_range = [0,2,0.2],
        )
            
        self.add(axes, stream_lines)
        stream_lines.start_animation(warm_up=False, flow_speed=1, time_width = 0.5)
        self.wait(1)
        self.play(stream_lines.end_animation())

class ContinuousMotion(Scene):
    def construct(self):
        a = 5
        stream_lines = StreamLines(van_der_pol,
        x_range = [-2*a, 2*a, a/4],
        y_range = [-a, a, a/4], stroke_width=3, max_anchors_per_line=30)
        bounding_area = Rectangle(
            width = 2*a, 
            height = 2*a,
            color = WHITE)
        ax = Axes(
            x_range = [-5*a, 5*a, a],
            y_range = [-5*a, 5*a, a]
        ).add_coordinates()
        circle = Circle().shift(LEFT)
        
        axes = NumberPlane(
            x_range = [-2*a, 2*a, 1],
            y_range = [-a, a, 1]
        )
        self.add(axes, stream_lines)
        stream_lines.start_animation(warm_up=False, flow_speed=1.5)
        self.wait(stream_lines.virtual_time / stream_lines.flow_speed)

class ContinuousMotion1(Scene):
    def construct(self):
        func_rot = lambda pos: - pos[1] * RIGHT + pos[0] * UP 
        func_dil = lambda pos: pos[0] * RIGHT + pos[1] * UP
        func_lox = lambda pos: (pos[0] - pos[1]) * RIGHT + (pos[0] + pos[1]) * UP
        stream_lines = StreamLines(func_lox, stroke_width=3, max_anchors_per_line=30)
        self.add(stream_lines)
        stream_lines.start_animation(warm_up=True, flow_speed=1)
        self.wait(5)