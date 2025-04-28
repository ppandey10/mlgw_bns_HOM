import math
import numpy as np

# Helper function to compute factorial (Python's math.factorial is used)
def factorial(n):
    return math.factorial(n)

# Function to compute the Wigner d-function d^l_{m,s}(i)
def wigner_d_function(l, m, s, i):
    """
    Compute the Wigner d-function d^l_{m,s}(i) as described in (II.8) of 
    https://arxiv.org/pdf/0709.0093.pdf.

    Parameters:
    l (int): upper index
    m (int): first lower index
    s (int): second lower index
    i (float): argument of the Wigner d-function (angle in radians)

    Returns:
    float: The value of the Wigner d-function d^l_{m,s}(i)
    """
    
    # Compute half-angle cos and sin for the given input
    costheta = math.cos(i * 0.5)
    sintheta = math.sin(i * 0.5)
    
    # Normalization factor: sqrt( (l+m)! * (l-m)! * (l+s)! * (l-s)! )
    norm = math.sqrt(factorial(l + m) * factorial(l - m) * factorial(l + s) * factorial(l - s))
    
    # Set the bounds for summation: ki and kf are the range of summation indices
    ki = max(0, m - s)
    kf = min(l + m, l - s)
    
    # Initialize the Wigner d-function value
    dWig = 0.0
    
    # Loop over k from ki to kf, summing the terms of the series
    for k in range(ki, kf + 1):
        # Compute the current term of the summation
        div = 1.0 / (factorial(k) * factorial(l + m - k) * factorial(l - s - k) * factorial(s - m + k))
        
        # Accumulate the terms: (-1)^k * (cos(theta))^(2l+m-s-2k) * (sin(theta))^(2k+s-m)
        dWig += div * (pow(-1, k) * pow(costheta, 2 * l + m - s - 2 * k) * pow(sintheta, 2 * k + s - m))
    
    # Return the final value: norm * sum of the series
    return norm * dWig

def spinsphericalharm(s, l, m, phi, i):
    """
    Compute the spin-weighted spherical harmonics Y_{lm}^s(phi, i).
    
    Parameters:
    s (int): Spin weight
    l (int): Multipolar index l
    m (int): Multipolar index m
    phi (float): Azimuthal angle (radians)
    i (float): Polar angle (radians)
    
    Returns:
    tuple: (rY, iY) where rY is the real part and iY is the imaginary part of the spin-weighted spherical harmonic
    """
    
    # Check for valid (l, m) values
    if l < 0 or m < -l or m > l:
        raise ValueError("Invalid (l,m) values in spinsphericalharm")
    
    # Compute the normalization constant c
    c = (-1.0)**(-s) * math.sqrt((2.0 * l + 1.0) / (4.0 * math.pi))
    
    # Compute the Wigner d-function value
    dWigner = c * wigner_d_function(l, m, -s, i)
    
    # Compute the real part of the spin-weighted spherical harmonic
    rY = math.cos(m * phi) * dWigner
    
    # Compute the imaginary part of the spin-weighted spherical harmonic
    iY = math.sin(m * phi) * dWigner
    
    # Return the real and imaginary parts
    return rY, iY

def unwrap_euler(p):
    size = len(p)
    if size < 1:
        return p  # Return the original array if size is less than 1

    dphi = 0.0
    corr = 0.0

    prev = p[0]
    delta = p[1] - p[0]

    for j in range(1, size):
        # Setting current data point
        p[j] += corr
        curr = p[j]

        # Check if decreasing too much - adding 2Pi
        if (curr < prev - np.pi) and (curr - prev < delta - np.pi):
            dphi = 2 * np.pi

        # Check if increasing too much - removing 2Pi
        if (curr > prev + np.pi) and (curr - prev > delta + np.pi):
            dphi = -2 * np.pi

        # Adding corrections
        corr += dphi
        p[j] += dphi

        # Resetting for next iteration
        prev = p[j]
        delta = p[j] - p[j - 1]
        dphi = 0.0

    return p  # Return the unwrapped array