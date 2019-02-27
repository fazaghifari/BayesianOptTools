import numpy as np

def rescale(xin,lb1,ub1,lb2,ub2):
    """
    :param xin: original input value
    :param lb1: lower bound input value
    :param ub1: upper bound input value
    :param lb2: lower bound output value
    :param ub2: upper bound output value
    :return: scaled number
    """
    if lb1 > ub1 or lb2 > ub2:
        raise ValueError("lower bound must be lower than upper bound")
    x = np.zeros(shape=[len(xin)])
    for ii in range(0,len(xin)):
        x[ii] = lb2 + (xin[ii]-lb1)*(ub2-lb2)/(ub1-lb1)
    return x