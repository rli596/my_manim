### 
#
# sg_utils.py
#
# utilities for sine-gordon animations
#
###

'''
Dependencies
'''
import numpy as np

'''
Functions
'''

def tractrix(t):
    '''
    Parameters
    ----------
    t (float): curve parameter
    
    Returns
    -------
    X (array): curve coordinate
    '''

    return np.array([1/np.cosh(t), t - np.tanh(t)])

def tractroid(u,v):
    '''
    Parameters
    ----------
    u, v (float): surface parameters
    
    Returns
    X (array): surface coordinate
    '''

    return np.array([tractrix(u)[0] * np.cos(v), tractrix(u)[0] * np.sin(v), tractrix(u)[1]])

def dini_surface(u,v,a,b):
    '''
    Parameters
    ----------
    u, v (float): surface parameters
    a, b (float): parameters
    
    Returns
    X (array): surface coordinate
    '''
    return np.array([a * np.cos(u) * np.sin(v), a * np.sin(u) * np.sin(v), a * (np.cos(v) + np.log(np.tan(v/2))) + b * u])

def pseudosphere_homotopy(x,y,z,t):
    '''
    Parameters
    ----------
    x,y,z,t (float): spatial coordinates and time parameter
    
    Returns
    -------
    X_dash (array): output coordinates

    Description
    -----------
    Transformation which turns tractroid into dini surface
    '''
    arg = np.angle(complex(x,y))
    return np.array([x, y, z + 0.1 * arg * t])

def breather_surface(u,v,a):
    '''
    Parameters
    ----------
    u, v (float): surface parameters
    a (float): parameter
    
    Returns
    -------
    X (array): surface coordinate
    '''
    w = np.sqrt(1 - a**2)
    denom = a * ((w * np.cosh(a * u))**2 + (a * np.sin(w * v))**2)

    x = 2 * w * np.cosh(a * u) * (-(w * np.cos(v) * np.cos(w * v)) - np.sin(v) * np.sin(w * v)) / denom
    y = 2 * w * np.cosh(a * u) * (-(w * np.sin(v) * np.cos(w * v)) + np.cos(v) * np.sin(w * v)) / denom
    z = -u + 2 * w **2 * np.cosh(a * u) * np.sinh(a * u) / denom
    return np.array([x,y,z])

def one_soliton(x):
    '''
    Parameters
    ----------
    x (float): line parameter
    
    Returns
    -------
    phi (float): function value
    '''
    phi = 4 * np.arctan(np.exp(x))
    return phi

def boosted_one_soliton(x,t, rapidity = -1):
    '''
    Parameters
    ----------
    x, t (float): arguments
    rapidity (float): rapidity
    
    Returns
    -------
    phi (float): function value
    '''
    phi = one_soliton( np.cosh(rapidity) * x + np.sinh(rapidity) * t)
    return phi

def breather(x,t, alpha = np.pi/3):
    '''
    Parameters
    ----------
    x, t (float): arguments
    alpha (float): parameter
    
    Returns
    -------
    phi (float): function value
    '''
    C = np.cos(alpha)
    S = np.sin(alpha)
    phi = 4 * np.arctan((S * np.cos(C * t)) / (C * np.cosh(S * x)))
    return phi

def two_soliton(x,t, beta = 1, m = 2):
    '''
    Parameters
    ----------
    x, t (float): arguments
    beta, m (float): parameter
    
    Returns
    -------
    phi (float): function value
    '''
    phi = 4 * np.arctan((beta * np.sinh(beta * m * x)) / (np.cosh(beta * m * t)))
    return phi

def kink_antikink(x,t, rapidity = 1):
    '''
    Parameters
    ----------
    x, t (float): arguments
    alpha (float): parameter
    
    Returns
    -------
    phi (float): function value
    '''
    C = np.cosh(rapidity)
    S = np.sinh(rapidity)
    T = np.tanh(rapidity)
    phi = 4 * np.arctan((T * np.cosh(C * x)) / (np.sinh(S * t)))
    return phi

def soliton_homotopy(x,y,z,t, func = boosted_one_soliton, t_range = [-40, 40]):
    '''
    Parameters
    ----------
    x,y,z,t (float): homotopy parameters
    func (function): function of x,t which solves sine-Gordon equation
    rapidity (float): rapidity
    
    Returns
    -------
    '''
    t_dash = t * (t_range[1] - t_range[0]) + t_range[0]

    C = np.cos(func(x,t_dash))
    S = np.sin(func(x,t_dash))
    R = np.array([[C, -S],
                  [-S, C]])
    X = x
    Y, Z = np.dot(R, np.array([y,z]))
    return np.array([X, Y, Z])

def soliton_plane_homotopy(x,y,z,t, func = boosted_one_soliton, t_range = [-20, 20]):
    '''
    Parameters
    ----------
    x,y,z,t (float): homotopy parameters
    func (function): function of x,t which solves sine-Gordon equation
    rapidity (float): rapidity
    
    Returns
    -------
    '''
    t_dash = t * (t_range[1] - t_range[0]) + t_range[0]

    X = x
    Y = func(x, t_dash)
    Z = 0
    return np.array([X, Y, Z])

'''
Testing
'''
if __name__ == "__main__":
    print(tractroid(0,0))