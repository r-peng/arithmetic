import numpy as np
import amp

n = 3
d = 10
p = 10
rep = 'eye'
rep = 'zero'
thresh = 1e-12
epsilon = 0.3
x = np.random.rand(d,n)
print(x.dtype)
old = []
for i in range(n):
    old.append(amp._input_tt(x[:,i],rep=rep,thresh=thresh))
old_ = [x[0,i] for i in range(n)]
tmp = np.zeros(d)
tmp[0] = 1
ik = [tmp for i in range(n)]
a = np.random.rand(p+1)

def _powers(y,p):
    powers = [y]
    for i in range(2,p+1):
        powers.insert(0,powers[0]*y)
    return powers
def _poly(y):
    n = len(a)-1
    powers = _powers(y,n)
    out = powers[0]*a[0]
    for i in range(1,n):
        out += powers[i]*a[i]
    return out + a[-1]
def _layer(ins,w):
    Li = []
    for i in range(n):
        yi = 0.0
        for j in range(n):
            yi += ins[j]*w[i,j]
        yi = _poly(yi)
        Li.append(yi)
    return Li
def _input_layer(old, old_, w):
    new = []
    for i in range(n):
        yi = []
        for j in range(n):
            yi.append(amp._scalar_mult(old[j],w[i,j],thresh))
        yi = amp._add_input_all(yi,thresh)
        yi = amp._poly(yi,a,rep,thresh)
        new.append(yi)
    new_ = _layer(old_,w)
    err = 0.0
    for i in range(n):
        out = amp._contract(new[i],ik)
        err += abs(new_[i]-out[0,0])
        print(new_[i],out[0,0])
        print(amp._get_bdim(new[i]))
    print(err)
    return new, new_
def _hidden_layer(old, old_, w):
    new = []
    for i in range(n):
        yi = amp._scalar_mult(old[0],w[i,0],thresh)
        for j in range(1,n):
            tmp = amp._scalar_mult(old[j],w[i,j],thresh)
            yi = amp._add_node(yi,tmp,thresh)
        out = amp._contract(yi,ik)
        print(out[0,0])
        new.append(yi)
    for i in range(n):
        new[i] = amp._poly(new[i],a,rep,thresh)
    new_ = _layer(old_,w)
    err = 0.0
    for i in range(n):
        out = amp._contract(new[i],ik)
        err += abs(new_[i]-out[0,0])
        print(new_[i],out[0,0])
        print(amp._get_bdim(new[i]))
    print(err)
    return new, new_
print('########### L=1 ##############')
w = np.random.rand(n,n)*epsilon
old, old_ = _input_layer(old,old_,w)
print('############## L=2 #######################')
w = np.random.rand(n,n)*epsilon
old, old_ = _hidden_layer(old,old_,w)
print('############## L=3 #######################')
w = np.random.rand(n,n)*epsilon
old, old_ = _hidden_layer(old,old_,w)
print('############## L=4 #######################')
w = np.random.rand(n,n)*epsilon
old, old_ = _hidden_layer(old,old_,w)
