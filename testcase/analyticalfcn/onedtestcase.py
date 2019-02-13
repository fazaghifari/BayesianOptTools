# source : http://infinity77.net/global_optimization/test_functions_1d.html

import numpy as np

def case10(x):
    #bound [0,10]
    f = -x * np.sin(x)
    return f