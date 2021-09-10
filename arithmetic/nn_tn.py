import numpy as np
import quimb.tensor as qtn
ADD = np.zeros((2,)*3)
ADD[0,0,0] = ADD[0,1,1] = ADD[1,0,1] = 1.0
def get_CP(d):
    out = np.zeros((d,)*3)
    for i in range(d):
        out[i,i,i] = 1.0
    return out
