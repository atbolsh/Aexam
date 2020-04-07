import numpy as np

def phi(w,b):
    """Computes phase.
    w = \omega, b = \gamma
    Both can be numpy arrays, for convenience."""
    return np.arctan2(2*w*b,(1-w**2))

def amplitude(w, b):
    """Computes amplitude^2.
    w = \omega, b = \gamma
    Both can be numpy arrays, for convenience."""
    return 1.0/np.sqrt(4*(w**2)*(b**2) + (1 - w**2)**2)

def datagen(muW, muB, stdW=0.1, stdB=0.1, size=10000, eps = 1e-3):
    """Creates the data and the values,
    eps is padding away from the b = 0 line;
    default is 1e-3."""
    # Frequencies
    w = np.random.randn(size, 1)*stdW + muW
    w = np.abs(w) # No need to deal with the symmetric case.
    # Damping coefficients
    b = np.random.randn(size, 1)*stdB + max(muB - eps, 0)
    b = np.abs(b) + eps # We are staying positive and away from the catastrophe
    b = np.clip(b, eps, 1.0) # Let's stick with the underdamped case.
    # Results
    P = phi(w, b)
    A = amplitude(w, b)
    # Concatenation
    z = np.concatenate((w, b), axis=1)
    o = np.concatenate((P, A), axis=1)
    return z, o

def log_datagen(muW, muB, stdW=0.1, stdB=0.1, size=10000, eps = 1e-3):
    # Frequencies
    lw = np.random.randn(size, 1)*stdW + muW
    # Damping coefficients
    lb = np.random.randn(size, 1)*stdB + muB
    # Results
    P = phi(np.exp(lw), np.exp(lb)) #Since phase is bounded, no need to be concerned here at all
    lA = np.log(amplitude(np.exp(lw), np.exp(lb)))
    # Concatenation
    z = np.concatenate((lw, lb), axis=1)
    o = np.concatenate((P, lA), axis=1)
    return z, o


