import numpy as np

def styb(x):
    d = len(x)
    sum = 0
    for ii in range (0,d):
        xi = x[ii]
        new = xi**4 - 16*xi**2 + 5*xi
        sum = sum + new
    y = sum/2
    return (y,0,0)

def branin (x):
    a = 5.1/(4 * (np.pi)**2 )
    b = 5/ np.pi
    c = (1-(1/(8 * np.pi)))

    f = (x[1] - a*x[1] + b*x[0] - 6)**2 + 10*(c* np.cos(x[0]) +1)
    return (f,0,0)
