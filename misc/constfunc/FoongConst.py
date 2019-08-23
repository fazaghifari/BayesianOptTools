import numpy as np
def foongconst(xnext):
    # THIS FUNCTION IS NOT GENERAL, FOR NOW IT'S ONLY FOR FOONG'S CASE
    consttable = np.loadtxt('../misc/constfunc/FoongConstraint.csv', delimiter=',')
    thetaround = np.floor(xnext[1]*2)/2
    result = np.where(consttable[:,0] == thetaround)
    index = result[0][0]
    betamax = consttable[index,2]
    if xnext[1] == 11.41:
        if xnext[2] <= 60.9564:
            coeff = 1
        else:
            coeff = 0
    else:
        if xnext[2] <= betamax:
            coeff = 1
        else:
            coeff = 0

    return coeff