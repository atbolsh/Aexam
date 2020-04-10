import numpy as np
import sklearn as skl
from sklearn.preprocessing import PolynomialFeatures
from numpy.linalg import lstsq

def total_size(shape):
    p = 1
    for dim in shape:
        p = p*dim
    return(p)

def polyeval(w, b, params, deg):
    s = w.shape
    size = total_size(s)
    pairs = np.concatenate((w.reshape((size, 1)), b.reshape((size, 1))), axis=1) 
    o = PolynomialFeatures(degree=deg)
    x_ = o.fit_transform(pairs)
    r = np.matmul(x_, params)
    return r.reshape(s)

def fitpoly(w, b, res, deg):
    s = w.shape
    size = total_size(s)
    pairs = np.concatenate((w.reshape((size, 1)), b.reshape((size, 1))), axis=1) 
    y = res.reshape(size)
    o = PolynomialFeatures(degree=deg)
    x_ = o.fit_transform(pairs)
    params, _, _, _ = lstsq(x_, y)
    return params

def params_to_deg(params):
    s = total_size(params.shape)
    return ((-1 + np.sqrt(1 + 8*s))/2) -1 #Inverse triangular number formula, minus 1

def deg_to_param_num(deg):
    return (deg+1)*(deg+2)/2 #Triangle number closed formula for (deg+1)


