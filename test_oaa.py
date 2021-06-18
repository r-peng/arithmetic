import numpy as np
import math
import rus, oaa
epsilon = 0.1
xs = [np.random.rand()*epsilon for i in range(2)]
R = rus._Rys(xs)
R = rus._gb(R)
for i in range(len(xs)):
    print(R[i,...])
