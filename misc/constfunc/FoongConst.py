import numpy as np
def tabulatedconst(ConstInfo,xnext):
    # THIS FUNCTION IS NOT GENERAL, FOR NOW IT'S ONLY FOR POTSAWAT'S CASE
    consttable = ConstInfo["constraint"]
    thetaround = np.floor(xnext[1]*2)/2
    result = np.where(consttable[:,0] == thetaround)
    index = result[0][0]
    betamax = consttable[index,1]
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