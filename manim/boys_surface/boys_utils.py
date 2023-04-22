'''
boys_utils.py

Utility functions for visualization of Boys surface
'''

# Dependencies
import numpy as np

# Functions
def hemisphere_to_disc(n):
    '''
    Parameters
    ----------
    n (array): 3D unit vector with z < 0
    
    Returns
    -------
    x (array): Plane vector with radius < 1
    
    Description
    -----------
    Maps the Southern hemisphere to the unit disc through stereographic projection
    '''
    x,y,z = n
    return np.array([x/(1-z), y/(1-z),0])

def disc_to_hemisphere(x):
    '''
    Parameters
    ----------
    x (array): Plane vector with radius < 1
    
    Returns
    -------
    n (array): 3D unit vector with z < 0
    '''
    X, Y, _ = x
    return np.array([2 * X / ( 1 + X**2 + Y**2), 2 * Y / (1 + X**2 + Y**2), (X**2 + Y**2 - 1)/(1 + X**2 + Y**2)])

def BK_g_functions(x):
    '''
    Parameters
    ----------
    x (array): Plane vector in unit disc
    
    Returns
    -------
    g (array): g = (g1,g2,g3) functions appearing in BK parametrization
    '''
    w = complex(x[0], x[1])

    g1 = -3/2 * np.imag(w * (1 - w**4) / (w**6 + np.sqrt(5) * w**3 - 1))
    g2 = -3/2 * np.real(w * (1 + w**4) / (w**6 + np.sqrt(5) * w**3 - 1))
    g3 = np.imag((1 + w**6) / (w**6 + np.sqrt(5) * w**3 - 1)) - 1/2

    return np.array([g1, g2, -g3])

def BK_param(x):
    '''
    Parameters
    ----------
    x (array): Plane vector in unit disc
    
    Returns
    -------
    X (array): Point on Boys surface
    '''
    g1, g2, g3 = BK_g_functions(x)
    g = BK_g_functions(x)
    denom = g1**2 + g2**2 + g3**2
    X = g/denom

    return X

def rotation_RP2_hemisphere(R,n):
    '''
    Parameters
    ----------
    R (array): Rotation matrix
    n (array): unit vector with z < 0

    Returns
    -------
    N (array): rotated unit vector, with z < 0
    '''
    N = np.dot(R, n)
    if N[-1] > 0:
        N = - N
    return N

def rotation_RP2_disc(R,x):
    '''
    Parameters
    ----------
    R (array): Rotation matrix
    x (array): vector in disc
    
    Returns
    -------
    X (array): vector in disc
    '''
    n = disc_to_hemisphere(x)
    N = rotation_RP2_hemisphere(R, n)
    X = hemisphere_to_disc(N)
    return X