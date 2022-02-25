####
# manimconfig.py
#
# To familiarise myself with ManimConfig.
# Following https://docs.manim.community/en/stable/tutorials/configuration.html
####

from manim import *
'''
config.background_color = WHITE # one way to set the background using attribute syntax
config['background_color'] = WHITE # using dictionary syntax

Camera().background_color
config.background_color = RED # 0xfc6255
Camera().background_color

config.frame_height
'''

class ShowScreenResolution(Scene):
    def construct(self):
        pixel_height = config["pixel_height"]  #  1080 is default
        pixel_width = config["pixel_width"]  # 1920 is default
        frame_width = config["frame_width"]
        frame_height = config["frame_height"]
        self.add(Dot())
        d1 = Line(frame_width * LEFT / 2, frame_width * RIGHT / 2).to_edge(DOWN)
        self.add(d1)
        self.add(Text(str(pixel_width)).next_to(d1, UP))
        d2 = Line(frame_height * UP / 2, frame_height * DOWN / 2).to_edge(RIGHT)
        self.add(d2)
        self.add(Text(str(pixel_height)).next_to(d2, LEFT))

'''
Using manim on the command line

manim -o myscene -i -n 0,10 -c WHITE <file.py> SceneName

Flags:
-o: output file name
-i: gif instead of mp4
-n: only the animations given in the range
-c: background colour
'''