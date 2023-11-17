'''
ising.py

manim animations of Ising model
'''

# Dependencies
import numpy as np
import random as rnd
from manim import *
from ising_utils import *

'''
Scenes
'''

class Ising(Scene):
    def construct(self):

        # Mobjects
        config_mob = VGroup()
        L = 10
        config = np.random.choice([-1, 1], size=(L,L), p=[.5, .5])
        for i in range(L):
            for j in range(L):
                if config[i,j] == -1:
                    config_mob.add(Dot(np.array([i-L/2,j-L/2,0])/2, color="RED"))
                if config[i,j] == 1:
                    config_mob.add(Dot(np.array([i-L/2,j-L/2,0])/2, color="BLUE"))

        # Animations
        self.add(config_mob)

        N = L**2
        T = 1
        beta = 1/T

        for i in range(L):
            for j in range(L):
                # Generate random lattice point
                a = np.random.randint(0, L)
                b = np.random.randint(0, L)
                
                s = config[a,b] # site
                nb = config[(a+1)%L,b] + config[a,(b+1)%L] + config[(a-1)%L,b] + config[a,(b-1)%L] # neighbours
                cost = 2*s*nb
                
                if cost < 0:
                    s *= -1
                elif rnd.random() < np.exp(-cost*beta):
                    s *= -1
                config[a, b] = s
                if config[a,b] == -1:
                    newDot = Dot(np.array([a-L/2,b-L/2,0])/2, color="RED")
                if config[a,b] == 1:
                    newDot = Dot(np.array([a-L/2,b-L/2,0])/2, color="BLUE")
                self.play(FadeIn(newDot))