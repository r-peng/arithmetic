import numpy as np
import quimb.tensor as qtn
import arithmetic.utils as utils
import itertools
np.set_printoptions(suppress=True,linewidth=100)
def morse(ri,rj,De=1.0,a=1.0,re=1.0):
    r = np.linalg.norm(ri-rj)
    return De*(1.0-np.exp(-a*(r-re)))**2
def get_tn3(r,beta,v_params,regularize=True):
    # regularize: set the largest value of each tensor to 1
    N,g,_ = r.shape
    tn = qtn.TensorNetwork([])
    for i in range(N):
        for j in range(i+1,N):
            data = np.zeros((g,)*2)
            for k in range(g):
                for l in range(g):
                    hij = morse(r[i,k,:],r[j,l,:],**v_params)
                    data[k,l] = np.exp(-beta*hij)
            if regularize:
                data_max = np.amax(data)
                data /= data_max
                expo = np.log10(data_max)
            else:
                expo = 0.0
            inds = 'r{}'.format(i),'r{}'.format(j)
            tags = set(inds).union({'exp'})
            tn.add_tensor(qtn.Tensor(data=data,inds=inds,tags=tags))
            tn.exponent = tn.exponent + expo
    for i in range(N):
        inds = ('r{}'.format(i),)
        tags = set(inds).union({'w'})
        tn.add_tensor(qtn.Tensor(data=np.ones(g),inds=inds,tags=tags))
    expo = tn.exponent
    tn.exponent = 0.0
    return tn,expo
def get_tn1(x,y,z,beta,v_params,regularize=True):
    N,g = x.shape
    ls = list(itertools.product(range(g),repeat=3))
    tn = qtn.TensorNetwork([])
    coords = ['x','y','z']
    for i in range(N):
        for j in range(i+1,N):
            data = np.zeros((g,)*6)
            for (xi,yi,zi) in ls:
                ri = np.array([x[i,xi],y[i,yi],z[i,zi]])
                for (xj,yj,zj) in ls:
                    rj = np.array([x[j,xj],y[j,yj],z[j,zj]])
                    hij = morse(ri,rj,**v_params)
                    data[xi,yi,zi,xj,yj,zj] = np.exp(-beta*hij)
            if regularize:
                data_max = np.amax(data)
                data /= data_max
                expo = np.log10(data_max)
            else:
                expo = 0.0
            inds = [c+str(i) for c in coords]+[c+str(j) for c in coords] 
            tags = set(inds).union({'exp'})
            tn.add_tensor(qtn.Tensor(data=data.copy(),inds=inds,tags=tags))
            tn.exponent = tn.exponent + expo
    for i in range(N):
        for c in ['x','y','z']:
            inds = (c+str(i),)
            tags = set(inds).union({'w'})
            tn.add_tensor(qtn.Tensor(data=np.ones(g),inds=inds,tags=tags))
    expo = tn.exponent
    tn.exponent = 0.0
    return tn,expo
compress_simplify_gauge = utils.compress_simplify_gauge
delete_tn_from_disc = utils.delete_tn_from_disc
load_tn_from_disc = utils.load_tn_from_disc
write_tn_to_disc = utils.write_tn_to_disc
