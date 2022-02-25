
'''
Dependencies
'''

from scipy.linalg import expm, eig
import numpy as np
from datetime import datetime
import cmath

'''
Functions
'''
# Parametrised manifolds
def param_sphere(u, v):
    R = np.pi
    x = R * np.sin(v) * np.cos(u)
    y = R * np.sin(v) * np.sin(u)
    z = R * np.cos(v)
    return np.array([x,y,z])

def param_line(start, end, t): # Now not needed
    '''
    Parameters
    ----------
    start: np.array
        start point of line
    end: np.array
        end point of line
    t: double
        parameter
    Returns
    -------
    p: point parametrised by t
    '''
    return start * (1-t) + end * t
    
# Linear algebra calculations
def skew_to_vec(skew):
    '''
    Parameters
    ----------
    skew: np.array
        3x3 skew symmetric matrix

    Returns
    -------
    vec: np.array
        3D vector
    '''
    x = skew[2, 1]
    y = -skew[2, 0]
    z = skew[1, 0]
    return np.array([x,y,z])

def vec_to_skew(vec):
    '''
    Parameters
    ----------
    skew: np.array
        3x3 skew symmetric matrix

    Returns
    -------
    vec: np.array
        3D vector
    '''
    skew = np.array([[0, -vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1], vec[0], 0]])
    return skew

def conjugatem(A, Q):
    '''
    Parameters
    ----------
    A: np.array
        3x3 matrix
    Q: np.array
        3x3 invertible matrix

    Returns
    -------
    B: np.array
        3x3 matrix coming from A conjugated by Q
    '''
    Qinv = np.linalg.inv(Q)
    B = np.dot(Q, np.dot(A, Qinv))
    return B

def ax_angle_to_mat(p):
    '''
    Parameters
    ----------
    p: np.array
        3D vector representing an SO(3) matrix via the axis-angle map

    Returns
    -------
    R_p: np.array
        3x3 matrix 
    '''
    R_p_gen = vec_to_skew(p)
    R_p = expm(R_p_gen)
    return R_p

def pushforward_by_ax_angle(X, p):
    '''
    Parameters
    ----------
    X: np.array
        3D vector representing an element of the lie algebra, isomorphic to the
        tangent space at the identity
    p: np.array
        3D vector representing an SO(3) matrix via the axis-angle map

    Returns
    -------
    Y: np.array
        3D vector representing the pushed forward element of the lie algebra, 
        living in the tangent space at p
    '''
    R_p = ax_angle_to_mat(p)
    xi = vec_to_skew(X)
    eta = conjugatem(xi, R_p)
    Y = skew_to_vec(eta)
    return Y

def pushforward_by_subset(X_0, ps):
    '''
    Parameters
    ----------
    X_0: np.array
        3D vector representing an element of the lie algebra, isomorphic to the
        tangent space at the identity
    ps: np.array
        Set of 3D vectors representing an SO(3) matrices via the axis-angle map

    Returns
    -------
    Ys: np.array
        3D vector representing the pushed forward element of the lie algebra, 
        living in the tangent space at p
    '''
    Ys = np.array([pushforward_by_ax_angle(X_0, p) for p in ps])
    return Ys

def pushforward_point_vec_pairs(X_0, ps):
    '''
    Parameters
    ----------
    X_0: np.array
        3D vector representing an element of the lie algebra, isomorphic to the
        tangent space at the identity
    ps: np.array
        Set of 3D vectors representing an SO(3) matrices via the axis-angle map

    Returns
    -------
    pairs: np.array
        selection of points taken from section of the tangent bundle. This can
        be understood as representing the vector field
    '''
    pairs = np.array([pair for pair in zip(ps, pushforward_by_subset(X_0, ps))])
    return pairs

def pushforward_vector_field(pairs, R_vec):
    '''
    Parameters
    ----------
    pairs: np.ndarray
        collection of (p, V) pairs, representing evaluations of a vector
        field
    R_vec: np.array
        3D vector representing an element of SO(3)
    
    Returns
    -------
    pushed_pairs: np.ndarray
        pushed forward vector field   
    '''
    R = ax_angle_to_mat(R_vec)
    count = 0
    for pair in pairs:
        '''
        # Things get tricky here. 
        # We're given pairs (p, V) where p lives under the axis-angle map, and
        # V is a tangent vector under the axis-angle map.
        # 
        # We know what to do for V: 
        # V -> xi_V via vec_to_skew
        # xi_V -> xi_R*V via conjugation
        # xi_R*V -> R*V via skew_to_vec
        #
        # To get R*p we need to be clever.
        # p -> R(p) via ax_angle_to_mat, noting R(p) neq R*p.
        # R(p) -> RR(p) under left translation by R
        # Now to recover R*p, we diagonalise RR(p)
        # There will be an eigenvector of 1, and two complex conjugate eigenvectors.
        # The 1 eigenvector gives the axis, while the cx conjugate evecs give angle.
        '''
        V = pair[1]
        xi_V = vec_to_skew(V)
        xi_RV = conjugatem(xi_V, R)
        RV = skew_to_vec(xi_RV)
        pair[1] = RV

        p = pair[0]
        Rp = ax_angle_to_mat(p)
        RRp = np.dot(R, Rp)
        evals, evecs_transposed = eig(RRp)
        evecs = np.transpose(evecs_transposed)

        one_evals = [index for index, evalue in enumerate(evals) \
            if not np.iscomplex(evalue) and evalue > 0] # Find index of eval 1
        one_eval_index = one_evals[0]
        one_evec = evecs[one_eval_index]
        axis = one_evec

        cx_eval_index = (one_eval_index + 1) % 3 # Fix first cx index to be after 1
        angle = np.angle(evals[cx_eval_index])

        # To check evecs are right handed set, thus ensuring
        # RRp is a clockwise rotation by angle about axis
        triple_prod = np.dot(evecs[one_eval_index], \
            np.cross(evecs[(one_eval_index+1)%3], \
                evecs[(one_eval_index+2)%3]))
        if np.iscomplex(triple_prod):
            handedness = np.sign(np.imag(triple_prod))
        else:
            handedness = np.sign(np.real(triple_prod))
        q = handedness * angle * axis
        '''
        Debugging
        '''
        if True:
            count+=1
            
        '''End of debugging'''
        pair[0] = q

    pushed_pairs = pairs
    return pushed_pairs

def log(msg):
    '''
    Parameters
    ----------
    msg: string
        message to be logged
    '''


'''
Main
'''
# Testing
if __name__=='__main__':
    '''
    theta = np.pi
    p_R_x = np.array([theta, 0, 0])
    R_x = ax_angle_to_mat(p_R_x)
    print(R_x)
    xi_z = np.array([0,0,1])
    print(xi_z)
    eta = pushforward_by_ax_angle(xi_z, p_R_x)
    print(eta)
    '''
    ps = []
    z_num_vals = 6
    x_num_vals = 3
    theta_num_vals = 6
    # Add vectors to vgroup
    for z in np.linspace(-np.pi, np.pi*(1 - 2/z_num_vals), z_num_vals):
        ps.append([0, 0, z])
    for z in np.linspace(-np.pi, np.pi*(1 - 2/z_num_vals), z_num_vals):
        for x in np.linspace(np.pi/3, np.sqrt(np.pi**2 - z**2), x_num_vals):
            for theta in np.linspace(0, 2*np.pi*(1-1/theta_num_vals), \
                theta_num_vals):
                ps.append([x*np.cos(theta), x*np.sin(theta), z])
    ps = np.array(ps)

    X_0 = np.array([0,0,1])
    pairs = pushforward_point_vec_pairs(X_0, ps)

    R_vec = np.array([0,0,np.pi/3])
    pushed_pairs = pushforward_vector_field(pairs, R_vec)