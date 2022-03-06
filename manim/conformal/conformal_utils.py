###
# conformal_utils.py
#
# Utilities for conformal mappings on the celestial torus
###

'''
Dependencies
'''
import numpy as np
from scipy.linalg import expm

'''
Constants
'''
# Pauli matrices

ID2 = np.array([[1, 0],
                [0, 1]])
SIGMA_X = np.array([[0, 1],
                    [1, 0]])
SIGMA_Y = np.array([[0, -1],
                    [1, 0]]) * complex(0, 1)
SIGMA_Z = np.array([[1, 0],
                    [0, -1]])

SIGMA_MU = np.array([ID2, SIGMA_X, SIGMA_Y, SIGMA_Z])

# SO(2,2) generators

J_1 = np.array([[0, -1, 0, 0],
                [1,  0, 0, 0],
                [0,  0, 0, 0],
                [0,  0, 0, 0]])
J_2 = np.array([[0, 0, 0,  0],
                [0, 0, 0,  0],
                [0, 0, 0, -1],
                [0, 0, 1,  0]])
K_11 = np.array([[0, 0, 1, 0],
                 [0, 0, 0, 0],
                 [1, 0, 0, 0],
                 [0, 0, 0, 0]])
K_12 = np.array([[0, 0, 0, 1],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [1, 0, 0, 0]])
K_21 = np.array([[0, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 0]])
K_22 = np.array([[0, 0, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 0, 0],
                 [0, 1, 0, 0]])

'''
Functions
'''
# Parametrised surfaces

def embedded_torus(u,v):
    '''
    Parameters
    ----------
        u (float): parameter one for surface
        v (float): parameters two
    Returns
    -------
        x (array): coordinate in three-dimensional space
    Constants
    ---------
        R (float): major radius, of circle in x-y plane
        r (float): minor radius in plane of constant of angle around major circle
                    r < R
    '''
    R = 2
    r = 1
    v1 = R * np.array([np.cos(u), np.sin(u), 0])
    v2 = r * ( np.cos(v) * np.array([np.cos(u), np.sin(u), 0] + np.sin(v) * np.array([0, 0, 1])))
    x = v1 + v2
    return x

# For transformations

def spacetime_to_angle(x):
    '''
    Parameters
    ----------
        x (array): (1,1) vector
    Returns
    -------
        thetas (array): pair of angles corresponding to point on celestial torus
    '''
    theta_1 = 2 * np.arctan(x[0])
    theta_2 = 2 * np.arctan(x[1])
    thetas = np.array([theta_1, theta_2])
    return thetas

def angle_to_null_celestial_torus(thetas):
    '''
    Parameters
    ----------
        thetas (array): pair of angles corresponding to point on celestial torus
    Returns
    -------
        x (array): (2,2) vector on celestial torus
    '''
    x = np.array([np.cos(thetas[0]), np.sin(thetas[0]), np.cos(thetas[1]), 
    np.sin([thetas[1]])])
    return x

def null_celestial_torus_to_angle(x):
    '''
    Parameters
    ----------
        x (array): (2,2) vector on celestial torus
    Returns
    -------
        thetas (array): pair of angles corresponding to point on celestial torus
    '''
    exp_1 = complex(x[0], x[1])
    exp_2 = complex(x[2], x[3])
    theta_1 = np.angle(exp_1)
    theta_2 = np.angle(exp_2)
    thetas = np.array([theta_1, theta_2])
    return thetas

def projective_celestial_action(x, Lambda):
    '''
    Parameters
    ----------
        x (array): (2,2) null vector 
        Lambda (array): SO(2,2) matrix
    Returns
    -------
        L_x (array): (2,2) transformed vector
    '''
    tl_magnitude = np.linalg.norm(x[0:2])
    L_x_unscaled = np.dot(Lambda, x)
    L_x = L_x_unscaled / tl_magnitude
    return L_x
    
'''
Testing
'''
if __name__ == "__main__":
    x = np.array([1, 0, 1, 0])
    Lambda = expm(J_1 * np.pi/3 - J_2 * np.pi/3) 
    L_x = projective_celestial_action(x, Lambda)
    print(L_x)