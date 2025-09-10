
import d2l
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x

x = np.arange(0, 3, 0.1)
d2l.plot(x, [f(x), 2 * x - 3, 3 * x -1], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])