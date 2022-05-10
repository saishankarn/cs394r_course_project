import numpy as np
import matplotlib.pyplot as plt

def interp0(x, xp, yp):
    """Zeroth order hold interpolation w/ same
    (base)   signature  as numpy.interp."""

    def func(x0):
        if x0 <= xp[0]:
            return yp[0]
        if x0 >= xp[-1]:
            return yp[-1]
        k = 0
        while x0 > xp[k]:
            k += 1
        return yp[k-1]
    
    if isinstance(x,float):
        return func(x)
    elif isinstance(x, list):
        return [func(x) for x in x]
    elif isinstance(x, np.ndarray):
        return np.asarray([func(x) for x in x])
    else:
        raise TypeError('argument must be float, list, or ndarray')

def NetworkStrength(N: int, n_min: int, n_max: int, low: int, high: int):
    n = np.random.randint(n_min, n_max + 1)
    
    anchor_x = np.random.choice(N, n, replace=False)
    anchor_x.sort()
    anchor_x = np.insert(anchor_x, 0, 0)

    anchor_y = [np.random.randint(low, high + 1)]
    for i in range(len(anchor_x)-1):
        if anchor_y[-1] == low:
            if anchor_y[-1] == high:
                anchor_y.append(anchor_y[-1])
            else:
                anchor_y.append(anchor_y[-1] + 1)
        elif anchor_y[-1] == high:
            anchor_y.append(anchor_y[-1] - 1)
        else:
            anchor_y.append(np.random.choice([anchor_y[-1] - 1, anchor_y[-1] + 1]))
    
    anchor_y = np.array(anchor_y)
    
    X = np.arange(0, N + 1)
    Y = interp0(X, anchor_x, anchor_y)

    return(Y)

yd = NetworkStrength(1000, n_min = 0, n_max = 2, low = 0, high = 0)