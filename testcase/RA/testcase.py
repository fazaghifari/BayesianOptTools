import numpy as np
from testcase.RA.bridgetruss_case import trussbridge

def evaluate (X,type = "fourbranches"):
    if X.ndim == 1:
        X = np.array([X])
    nsample = np.size(X, 0)
    y = np.zeros(shape=[nsample, 1])
    if type.lower() == "fourbranches":
        for ii in range(0, nsample):
            y[ii,0] = fourbranches(X[ii, 0],X[ii, 1])
    elif type.lower() == "styblinski":
        for ii in range(0, nsample):
            y[ii, 0] = styb(X[ii, :])
    elif type.lower() == "branin":
        for ii in range(0, nsample):
            y[ii, 0] = branin(X[ii, :])
    elif type.lower() == "hidimenra":
        y = hidimenRA(X)
    elif type.lower() == "hidimenra2":
        y = hidimenRA2(X)
    elif type.lower() == "bridge":
        for ii in range(nsample):
            if ii % 50000 == 0 and ii != 0:
                print("eval number",ii)
            num_tri = 6
            Ediag = np.ones(shape=num_tri) * X[ii,1]
            Adiag = np.ones(shape=num_tri) * X[ii,3]
            Ebot = np.ones(shape=num_tri) * X[ii,0]
            Abot = np.ones(shape=num_tri) * X[ii,2]
            Etop = np.ones(shape=num_tri - 1) * X[ii,0]
            Atop = np.ones(shape=num_tri - 1) * X[ii,2]
            p = -X[ii,4:10]
            res = trussbridge(Ediag,Adiag,Ebot,Abot,Etop,Atop,p)
            y[ii, 0] = (res['uy'] + 0.1)
    else:
        raise NameError("Test case unavailable!")

    return y

def fourbranches (x1,x2,k=7):
    yall = np.zeros(shape=[4])
    yall[0] = 3 + 0.1 * (x1 - x2) ** 2 - ((x1 + x2) / np.sqrt(2))
    yall[1] = 3 + 0.1 * (x1 - x2) ** 2 + ((x1 + x2) / np.sqrt(2))
    yall[2] = (x1 - x2) + (k / np.sqrt(2))
    yall[3] = (x2 - x1) + (k / np.sqrt(2))
    y = np.min(yall)
    return y

def styb(x):
    """
    modified styblinski-tang
    """
    d = len(x)
    sum = 0
    for ii in range(0, d):
        xi = x[ii]
        new = xi ** 4 - 16 * xi ** 2 + 5 * xi
        sum = sum + new
    y = 50 - sum / 2
    return y

def branin (x):
    """
        modified branin
    """
    a = 5.1/(4 * (np.pi)**2 )
    b = 5/ np.pi
    c = (1-(1/(8 * np.pi)))

    f = -((x[1] - a*x[1] + b*x[0] - 6)**2 + 10*(c* np.cos(x[0]) +1) - 70)
    return f

def hidimenRA(x):

    n = np.size(x,axis=1)
    sigma = 0.2
    tempvar = n + 3*sigma*np.sqrt(n)

    f = tempvar - np.sum(x,axis=1).reshape(-1,1)
    return f

def hidimenRA2(x):

    n = np.size(x, axis=1)
    tempvar = 3 - x[:,-1]
    xtemp = x[:,:n-1]

    f = tempvar.reshape(-1,1) + 0.01 * np.sum(xtemp**2,axis=1).reshape(-1,1)
    return f