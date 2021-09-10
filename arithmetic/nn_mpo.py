import numpy as np
import quimb.tensor as qtn
ADD = np.zeros((2,)*3)
ADD[0,0,0] = ADD[0,1,1] = ADD[1,0,1] = 1.0
def get_CP(d):
    out = np.zeros((d,)*3)
    for i in range(d):
        out[i,i,i] = 1.0
    return out
def get_weighted_sum1(Xs,ws,b):
    # weighted sum
    arrays = []
    for i in range(len(Xs)):
        tmp = np.einsum('ipq,i->ipq',Xs[i],np.array([1.0,ws[i]))
        if i>0:
            tmp = np.einsum('ipq,ijk->jkpq',tmp,ADD)
        arrays.append(tmp)
    tmp = np.einsum('ijk,j->ik',ADD,np.array([1.0,b]))
    arrays.append(tmp)
    return arrays
