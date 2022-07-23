###
# lorentz_utils.py
#
# Utilities for Lorentz group maps
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

'''
Functions
'''
# Parametrised surfaces
def light_cone(u,v):
    '''
    Parameters
    ----------
        u (float): 
        v (float):
    Returns
    -------
        x (array):
    '''
    x = np.array([u*np.cos(v), u*np.sin(v), u])
    return x

# First for 1+2 dimensions
def vec_to_boost_2(v):
    '''
    Parameters
    ----------
        v (array): 2d vector giving direction of boost
    Returns
    -------
        Lambda (array): 3x3 matrix for Lorentz boost
    '''
    v1, v2 = v
    gen = np.array([[0, 0, v1],
                   [0, 0, v2],
                   [v1, v2, 0]])
    Lambda = expm(gen)
    return Lambda

def rescaling_2(x):
    '''
    Parameters
    ----------
        x (array): relativistic 3-vector
    Returns
    -------
        X (array): relativistic 3-vector rescaled to unit time
    '''
    x_1, x_2, t = x
    X = x/t
    return X

# Now for 3 + 1 dimensions
def vec_to_boost_3(v):
    '''
    Parameters
    ----------
        v (array): 2d vector giving direction of boost
    Returns
    -------
        Lambda (array): 4x4 matrix for Lorentz boost
    '''
    v1, v2, v3 = v
    gen = np.array([[0, v1, v2, v3],
                   [v1, 0, 0, 0],
                   [v2, 0, 0, 0],
                   [v3, 0, 0, 0]])
    Lambda = expm(gen)
    return Lambda

def lorentz_transform_spacelike_proj(x, Lambda):
    '''
    Parameters
    ----------
        x (array): spacelike part of null, positive directed 4-vector
        Lambda (array): 4x4 matrix of Lorentz transformation
    Returns
    -------
        L_x (array): spacelike part of Lorentz transformed null 4-vector
    '''
    t = np.linalg.norm(x)
    x_1, x_2, x_3 = x
    x_mu = np.array([t, x_1, x_2, x_3])
    L_x_mu = np.dot(Lambda, x_mu)
    L_x_0, L_x_1, L_x_2, L_x_3 = L_x_mu
    L_x = np.array([L_x_1, L_x_2, L_x_3])
    return L_x

def rescaling_3(x):
    '''
    Parameters
    ----------
        x (array): spatial components of null 4-vector
    Returns
    -------
        X (array): spatial components of null 4-vector rescaled to unit time
    '''
    x_1, x_2, x_3 = x
    t = np.linalg.norm(x) # Since x is null, and the 4-vector is positive directed, t is norm of x
    X = x/t
    return X

def conformal_transformation_1_2(x, Lambda):
    '''
    Parameters
    ----------
        x (array): null, positive directed 3-vector
        Lambda (array): 3x3 matrix of Lorentz transformation
    Returns
    -------
        lambda_L_x (array): spacelike part of Lorentz transformed null 4-vector rescaled to unit time
    '''
    L_x = np.dot(Lambda, x)
    lambda_L_x = L_x/L_x[-1]
    return lambda_L_x

def conformal_transformation_3_1(x, Lambda):
    '''
    Parameters
    ----------
        x (array): spacelike part of null, positive directed 4-vector
        Lambda (array): 4x4 matrix of Lorentz transformation
    Returns
    -------
        X (array): spacelike part of Lorentz transformed null 4-vector rescaled to unit time
    '''
    L_x = lorentz_transform_spacelike_proj(x, Lambda)
    X = rescaling_3(L_x)
    return X

def stereographic_projection(n):
    '''
    Parameters
    ----------
        n (array): point on sphere
    Returns
    -------
        X (array): point on stereographic plane
    '''
    x, y, z = n
    X = np.array([x/(1-z), y/(1-z), 0])
    return X

def stereographic_inverse(X):
    '''
    Parameters
    ----------
        X (array): point on stereographic plane
    Returns
    -------
        n (array): point on sphere
    '''
    X, Y, _ = X
    denom = 1 + X**2 + Y**2
    x = 2 * X / denom
    y = 2 * Y / denom
    z = (-1 + X**2 + Y**2) / denom
    n = np.array([x, y, z])
    return n

def map_from_matrix(S):
    '''
    Parameters
    ----------
        S (array): complex 2x2 matrix in SL(2;C)
    Returns
    -------
        f (function): complex function of one complex variable
    '''
    # Unpack values of S
    S_1, S_2 = S
    S_11, S_12 = S_1
    S_21, S_22 = S_2 

    f = lambda z: (S_11 * z + S_12)/(S_21 * z + S_22)
    return f

def real_transformation_from_cx_map(f):
    '''
    Parameters
    ----------
        f (function): Möbius map
    Returns
    -------
        F (function): Function which takes in a planar vector and returns a planar vector
    '''
    
    F = lambda x: np.array([np.real(f(complex(x[0], x[1]))),
                            np.imag(f(complex(x[0], x[1]))),
                            0])
    return F

def real_transformation_from_matrix(S):
    f = map_from_matrix(S)
    F = real_transformation_from_cx_map(f)
    return F

def conformal_transformation_mobius(x, S):
    '''
    Parameters
    ----------
        x (array): 3d vector
        S (array): complex 2x2 matrix in SL(2;C)
    Returns
    -------
        X (array): 3d vector
    '''
    F = real_transformation_from_matrix(S)
    y = stereographic_projection(x)
    Y = F(y)
    X = stereographic_inverse(Y)
    return X

def SL_to_SO(S):
    '''
    Parameters
    ----------
        S (array): complex 2x2 matrix in SL(2;C)
    Returns
    -------
        Lambda (array): real 4x4 matrix in SO(3,1)
    '''
    Lambda = np.zeros([4, 4])
    S = np.matrix(S)
    for mu in range(4):
        for nu in range(4):
            F = SIGMA_MU[nu]
            S_dagger = S.getH()
            C = np.dot(S, np.dot(F, S_dagger))
            F_inv = 1/2 * np.trace(np.dot(SIGMA_MU[mu], C))
            Lambda[mu, nu] = F_inv
    return Lambda
    
'''
Testing
'''
if __name__ == "__main__":
    a = complex(1, 0)
    S = np.array([[1, a],
                  [0, 1]])
    Lambda = SL_to_SO(S)
    print(Lambda)