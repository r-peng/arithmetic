import numpy as np
import quimb.tensor as qtn
import utils
ADD = np.zeros((2,)*3)
ADD[0,0,0] = ADD[1,0,1] = ADD[0,1,1] = 1.0
CP2 = np.zeros((2,)*3)
CP2[0,0,0] = CP2[1,1,1] = 1.0
def get_peps(xs):
    N,k,d = xs.shape
    CP = np.zeros((d,)*3)
    for i in range(d):
        CP[i,i,i] = 1.0
    arrays = []
    for i in range(k):
        rows = []
        for j in range(N):
            data = np.ones((2,d))
            data[1,:] = xs[j,i,:]
            if i>0:
                data = np.einsum('ir,rpq->ipq',data,CP)
            if j>0:
                data = np.einsum('i...,ijk->jk...',data,ADD)
            if j==N-1:
                data = data[:,1,...]
            rows.append(data)
        arrays.append(rows)
    return make_peps_with_legs(arrays)    

permute_1d = utils.permute_1d
contract_1d = utils.contract_1d
make_peps_with_legs = utils.make_peps_with_legs
trace_open = utils.trace_open
contract_from_bottom = utils.contract_from_bottom
contract = utils.contract
