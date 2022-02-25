###
# hopf/utils.py
#
# Useful functions for visualising the Hopf map
###

'''
Dependencies
'''
import numpy as np
import sys
from scipy.spatial.transform import Rotation

'''
Functions
'''
def real_4_to_cx_2(x):
    '''
    Parameters:
        x (array): array of four real numbers
    Returns:
        z (array): array of two complex numbers
    '''
    x_0 = x[0]
    y_0 = x[1]
    x_1 = x[2]
    y_1 = x[3]

    z_0 = complex(x_0, y_0)
    z_1 = complex(x_1, y_1)
    z = np.array([z_0, z_1])
    return z

def cx_2_to_real_4(z):
    '''
    Parameters:
        z (array): array of two complex numbers
    Returns:
        x (array): array of four real numbers
    '''
    z_0 = z[0]
    z_1 = z[1]
    x_0 = z_0.real
    y_0 = z_0.imag
    x_1 = z_1.real
    y_1 = z_1.imag
    x = np.array([x_0, y_0, x_1, y_1])
    return x
    
def hopf_map(z):
    '''
    Parameters:
        z (array): array of two complex numbers
    Returns:
        p (array): array of one complex number and one real number
    '''
    z_0 = z[0]
    z_1 = z[1]
    norm_sq = abs(z_0)**2 + abs(z_1)**2
    p_0_unnormed = 2 * z_0.conjugate() * z_1
    p_1_unnormed = abs(z_1)**2 - abs(z_0)**2
    p_0 = p_0_unnormed / norm_sq
    p_1 = p_1_unnormed / norm_sq
    p = np.array([p_0, p_1])
    return p

def stereog_proj(x):
    '''
    Parameters:
        x (array): array of four real numbers
    Returns:
        phi_N (array): array of three real numbers
    '''
    big_num = 1000
    w,z,x,y = x
    if w != 1:
        X = x/(1-w)
        Y = y/(1-w)
        Z = z/(1-w)
    else:
        X = big_num * x
        Y = big_num * y
        Z = big_num * z
    phi_N = np.array([X,Y,Z])
    return phi_N

def stereog_inv(X):
    '''
    Parameters:
        X (array): array of three real numbers
    Returns:
        x (array): array of four real numbers
    '''
    X,Y,Z = X
    R_sq = X**2 + Y**2 + Z**2
    # (1+w)/(1-w) = R_sq
    w = (R_sq - 1)/(R_sq + 1)
    x = X * (1-w)
    y = Y * (1-w)
    z = Z * (1-w)
    x = np.array([w,z,x,y])
    return x

def two_balls(x):
    '''
    Parameters:
        x (array): array of four real numbers
    Returns:
        X (array): array of three real numbers
    '''
    w, z, x, y = x
    if w < 0:
        X = x
        Y = y
        Z = -2 + z
    elif w >= 0:
        X = x
        Y = y
        Z = 2 + z
    X = np.array([X, Y, Z])
    return X

def SO3_proj(x):
    '''
    Parameters:
        x (array): array of four real numbers, vector on the 3-sphere
    Returns:
        X (array): array of three real numbers, axis-angle representation of 
        SU(2) as an SO(3) matrix
    '''
    w, x, y, z = x
    R = Rotation.from_quat([w,x,y,z])
    X = Rotation.as_rotvec(R)
    return X

def angle_to_3_vec(Omega):
    '''
    Parameters:
        Omega (array): array of two real numbers, (theta, phi), representing 
        a 2d angle
    Returns:
        n (array): array of three real numbers
    '''
    theta, phi = Omega
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    n = np.array([x, y, z])

    return n

def vec_to_angle(n):
    '''
    Parameters:
        n (array): array of three real numbers
    Returns:
        Omega (array): array of two real numbers, (theta, phi), representing 
        a 2d angle
    '''
    x, y, z = n
    phi = np.angle(complex(x,y))
    theta = np.arccos(z)
    Omega = np.array([theta, phi])
    return Omega

def angle_to_rep_4_vec(Omega):
    '''
    Parameters:
        Omega (array): array of two real numbers, (theta, phi), representing 
        a 2d angle
    Returns:
        x (array): array of four real numbers
    '''
    theta, phi = Omega
    z_0 = np.exp(complex(0, -phi/2)) * np.sin(theta/2)
    z_1 = np.exp(complex(0, phi/2)) * np.cos(theta/2)
    z = np.array([z_0, z_1])
    x = cx_2_to_real_4(z)
    return x

def rep_4_vec_orbit(x, phi_dash):
    '''
    Parameters:
        x (array): array of four real numbers
        phi_dash (float): phase to apply
    Returns:
        x_dash (array): array of four real numbers
    '''
    w = np.exp(complex(0, phi_dash/2))
    z = real_4_to_cx_2(x)
    z_dash = w * z
    x_dash = cx_2_to_real_4(z_dash)
    return x_dash

def fund_flow_stereog(X, phi_dash):
    '''
    Parameters:
        X (array): array of three real numbers
        phi_dash (float): phase to apply
    Returns:
        X_dash (array): array of three real numbers
    '''
    x = stereog_inv(X)
    x_dash = rep_4_vec_orbit(x, phi_dash)
    X_dash = stereog_proj(x_dash)
    return X_dash

# Parametrisation of Hopf spiral

def spiral_param(u, v):
    '''
    Parameters
    ----------
        u (float): First argument, varies as surface is created. Plays role of
        theta
        v (float): Second argument, fixed when surface is created. Plays role of 
        phi_dash
    Local constant
    --------------
        n (float): Number of times the spiral on S^2 wraps around the sphere
    Returns
    -------
        X (array): 3-dimensional array, representing stereographic projection of
        the spiral
    '''
    n = 10
    w = np.cos(v/2 - n * u) * np.sin(u/2)
    x = np.sin(v/2 - n * u) * np.sin(u/2)
    y = np.cos(v/2 + n * u) * np.cos(u/2)
    z = np.sin(v/2 + n * u) * np.cos(u/2)

    if w != 1:
        X = np.array([x/(1-w), y/(1-w), z/(1-w)])
    if w == 1:
        X = 1000*np.array([x,y,z])
    return X

'''
Testing
'''
if __name__ == "__main__":
    if sys.argv[1] == "test":
        x = np.array([0, 0, 0, 1])
        print(stereog_proj(x))
        print(stereog_inv(stereog_proj(x)))
    