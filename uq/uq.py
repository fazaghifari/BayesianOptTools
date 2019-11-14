import numpy as np

def meanstd (y):
    mean = np.mean(y)
    stdev = np.std(y)

    return (mean,stdev)