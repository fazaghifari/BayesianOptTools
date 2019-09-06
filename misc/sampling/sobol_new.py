import numpy as np
import pandas as pd


def sobol_points(n, d):
    """
    Sobol points generator based on graycode order.
    This function is translated from the original c++ program.
    Original c++ program: https://web.maths.unsw.edu.au/~fkuo/sobol/

    Args:
         n (int): number of points (cannot be greater than 2^32)
         d (int): number of dimension

     Return:
         point (nparray): 2-dimensional array with row as the point and column as the dimension.
    """
    # convert n and d into int in case n and d are float.
    n = int(n)
    d = int(d)

    soboldir = pd.read_csv('../misc/sampling/sobolcoeff.csv')

    # ll = number of bits needed
    ll = int(np.ceil(np.log(n)/np.log(2.0)))

    # c[i] = index from the right of the first zero bit of i
    c = np.zeros(shape=[n])
    c[0] = 1
    for i in range(1, n):
        c[i] = 1
        value = i
        while value & 1:
            value >>= 1
            c[i] += 1
    c = c.astype(int)

    # points initialization
    points = np.zeros(shape=[n, d])

    # ----- Compute the first dimension -----
    # Compute direction numbers v[1] to v[L], scaled by 2**32
    v = np.zeros(shape=[ll+1])
    for i in range(1, ll+1):
        v[i] = 1 << (32-i)
        v[i] = int(v[i])

    #  Evalulate x[0] to x[N-1], scaled by 2**32
    x = np.zeros(shape=[n])
    x[0] = 0
    for i in range(1, n):
        x[i] = int(x[i-1]) ^ int(v[c[i-1]])
        points[i, 0] = x[i]/(2**32)

    # Clean variables
    del v
    del x

    # ----- Compute the remaining dimensions -----
    for j in range(1, d):

        # read parameters from file
        dd = int(soboldir.iloc[j - 1]['d'])
        s = int(soboldir.iloc[j - 1]['s'])
        a = int(soboldir.iloc[j - 1]['a'])
        mm = soboldir.iloc[j - 1]['m_i']
        mm = mm.split()
        m = np.array([0]+mm).astype(int)

        # Compute direction numbers V[1] to V[L], scaled by 2**32
        v = np.zeros(shape=[ll+1])
        if ll <= s:
            for i in range(1, ll+1):
                v[i] = int(m[i]) << (32-i)

        else:
            for i in range(1, s+1):
                v[i] = int(m[i]) << (32-i)

            for i in range(s+1, ll+1):
                v[i] = int(v[i-s]) ^ (int(v[i-s]) >> s)
                for k in range(1, s):
                    v[i] = int(v[i]) ^ (((int(a) >> int(s-1-k)) & 1) * int(v[i-k]))

        # Evalulate X[0] to X[N-1], scaled by pow(2,32)
        x = np.zeros(shape=[n])
        x[0] = 0
        for i in range(1, n):
            x[i] = int(x[i-1]) ^ int(v[c[i-1]])
            points[i, j] = x[i]/(2**32)

        del m
        del v
        del x

    return points


if __name__ == "__main__":
    point = sobol_points(10, 3)
    print(point)
