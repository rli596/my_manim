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
def real_4_to_cx_2(x):
    '''
    Parameters
    ----------
    x (array): array of four real numbers

    Returns:
    --------
    z (array): array of two complex numbers

    Description
    -----------
    Converts array of 4 real numbers (x_0,y_0,x_1,y_1) to array of 2 complex numbers (x_0 + iy_0, x_1 + iy_1)
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
    Parameters
    ----------
    z (array): array of two complex numbers

    Returns
    -------
    x (array): array of four real numbers

    Description
    -----------
    Converts array of 2 complex numbers (z_0, z_1) to array of 4 real numbers (Re(z_0), Im(z_0), Re(z_1), Im(z_1))
    '''
    z_0 = z[0]
    z_1 = z[1]
    x_0 = z_0.real
    y_0 = z_0.imag
    x_1 = z_1.real
    y_1 = z_1.imag
    x = np.array([x_0, y_0, x_1, y_1])
    return x

def cx_2_to_SU_2(z):
    '''
    Parameters
    ----------
    z (array): array of two complex numbers, belonging to S^3
    
    Returns
    -------
    U (array): 2x2 matrix of complex numbers, element of SU(2)
    '''
    return np.array([[z[0], -z[1].conj()], [z[1], z[0].conj()]])

def SU_2_to_cx_2(U):
    '''
    Parameters
    ----------
    U (array): 2x2 matrix of complex numbers, element of SU(2)

    Returns
    -------
    z (array): array of two complex numbers, belonging to S^3
    '''
    row_1, row_2 = U
    alpha, _ = row_1
    beta, _ = row_2
    return np.array([alpha, beta])
    
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

def stereographic_projection_3d(n):
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

def stereographic_inverse_3d(X):
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

def S2_to_preimage(n):
    '''
    Parameters
    ----------
    n (array): array of three real numbers, with norm 1
    
    Returns
    -------
    circle (function: float -> array): parametrised circle for preimage of n
    '''

    spherical_angle = vec_to_angle(n)
    representative_pt = angle_to_rep_4_vec(spherical_angle)
    param_curve = lambda t: stereog_proj(rep_4_vec_orbit(representative_pt, 4 * np.pi * t)) # Normalisation of phase so when t = 1, goes through full circle

    return param_curve

def pauli_rot_homotopy(x,y,z,t):
    '''
    Parameters
    ----------
    x,y,z,t (float): spatial coordinates and time parameter
    
    Returns
    -------
    X_dash (array): output coordinates

    Description
    -----------
    Transformation of S^3 under stereographic projections when S^2 is rotated 
    '''
    X_stereo = np.array([x,y,z])
    X_4 = stereog_inv(X_stereo)
    xi = real_4_to_cx_2(X_4)
    U = cx_2_to_SU_2(xi)

    U_tfm = expm(complex(0,1) * np.pi/2 * t * SIGMA_X)
    
    U_dash = np.dot(U_tfm, U)
    xi_dash = SU_2_to_cx_2(U_dash)
    X_4_dash = cx_2_to_real_4(xi_dash)
    X_dash = stereog_proj(X_4_dash)
    return X_dash

def fibrewise_rot_homotopy(x,y,z,t):
    '''
    Parameters
    ----------
    x,y,z,t (float): spatial coordinates and time parameter
    
    Returns
    -------
    X_dash (array): output coordinates

    Description
    -----------
    Transformation of S^3 under stereographic projections under fibrewise rotation
    '''
    X_stereo = np.array([x,y,z])
    X_4 = stereog_inv(X_stereo)
    xi = real_4_to_cx_2(X_4)

    xi_dash = xi * np.exp(-complex(0,1) * 2 * np.pi/3 * t)
    X_4_dash = cx_2_to_real_4(xi_dash)
    X_dash = stereog_proj(X_4_dash)
    return X_dash

def funky_fibrewise_rot_homotopy(x,y,z,t):
    '''
    Parameters
    ----------
    x,y,z,t (float): spatial coordinates and time parameter
    
    Returns
    -------
    X_dash (array): output coordinates

    Description
    -----------
    Transformation of S^3 under stereographic projections under fibrewise rotation
    '''
    X_stereo = np.array([x,y,z])
    X_4 = stereog_inv(X_stereo)
    xi = real_4_to_cx_2(X_4)

    xi_dash = xi * np.exp(-complex(0,1) * 2 * np.pi/3 * t * x)
    X_4_dash = cx_2_to_real_4(xi_dash)
    X_dash = stereog_proj(X_4_dash)
    return X_dash

def apollonian_tori_homotopy(x,y,z,t):
    '''
    Parameters
    ----------
    x,y,z,t (float): spatial coordinates and time parameter
    
    Returns
    -------
    X_dash (array): output coordinates

    Description
    -----------
    Sends tori of apollonius to each other 
    '''
    z_xy = complex(x,y)
    r = abs(z_xy)
    theta = np.angle(z_xy)

    z_rz = complex(r,z)

    S_tfm = expm(t * SIGMA_X)
    row_1, row_2 = S_tfm
    a, b = row_1
    c, d = row_2
    
    z_rz_dash = (a * z_rz + b)/(c * z_rz + d)
    z_dash = z_rz_dash.imag
    r_dash = z_rz_dash.real

    x_dash = r_dash * np.cos(theta)
    y_dash = r_dash * np.sin(theta)

    return np.array([x_dash, y_dash, z_dash])

def z_boost_homotopy(x,y,z,t):
    '''
    Parameters
    ----------
    x,y,z,t (float): spatial coordinates and time parameter
    
    Returns
    -------
    X_dash (array): output coordinates

    Description
    -----------
    Applies z-boost to celestial sphere 
    '''
    Z = complex(x/(1-z), y/(1-z))
    Z_dash = np.exp(t) * Z

    X_dash = Z_dash.real
    Y_dash = Z_dash.imag
    R_dash = abs(Z_dash)

    z_dash = (R_dash**2 - 1)/(R_dash**2 + 1)
    x_dash = X_dash * (1-z_dash)
    y_dash = Y_dash * (1-z_dash)

    return np.array([x_dash, y_dash, z_dash])

# Parametrisations

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

def villarceau_torus(u,v,n):
    '''
    Parameters
    ----------
    u (float): Parameter for villarceau circle
    v (float): Rotation parameter for circle
    n (array): unit vector corresponding to point in base S^2

    Returns
    -------
    X (array): 3D position of point along torus
    '''
    unrotated_X = S2_to_preimage(n)(u)
    rotation_generator = np.array([[0,-1,0], [1,0,0], [0,0,0]])
    rotation_matrix = expm(2 * np.pi * rotation_generator * v)

    X = np.dot(rotation_matrix, unrotated_X)

    return X

def villarceau_scale_factor(v, r = 1, R = 2):
    '''
    Parameters
    ----------
    v (float): minor radius coordinate
    r (float): minor radius
    R (float): major radius

    Returns
    -------
    S (float): scale factor
    '''
    R_dash = np.sqrt(R**2 - r**2)
    return np.arccos((R_dash * np.sin(v))/(R + r * np.cos(v)))/(2 * np.pi)

def villarceau_cutaway(u,v):
    '''
    Parameters
    ----------
    u (float): major radius coordinate
    v (float): minor radius coordinate

    Returns
    -------
    X (array): coordinate of parametrization
    '''
    X = np.array([(2+ np.cos(v)) * np.sin(villarceau_scale_factor(v) * (u)), (2+ np.cos(v)) * np.cos(villarceau_scale_factor(v) * (u)), np.sin(v)])
    return X

def torus_min_maj(u,v,r,R):
    '''
    Parameters
    ----------
    u (float): major radius coordinate
    v (float): minor radius coordinate
    r (float): minor radius
    R (float): major radius

    Returns
    -------
    X (array): coordinate of parametrization
    '''
    return np.array([(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)])

'''
Testing
'''
if __name__ == "__main__":
    if sys.argv[1] == "test":
        x = np.array([0, 0, 0, 1])
        print(stereog_proj(x))
        print(stereog_inv(stereog_proj(x)))
    