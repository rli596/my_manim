'''
Dependencies
'''
import numpy as np

'''
Functions
'''
def isp(x):
    '''
    x (array): planar vector
    '''
    X, Y, Z = x
    R = np.sqrt(X**2 + Y**2)
    return 2* np.array([2 * X / (1 + R**2), 2 * Y / (1 + R**2), (R**2 - 1)/(1 + R**2)])

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
    X = 2 * np.array([x/(2-z), y/(2-z), 0])
    return X