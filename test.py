import numpy as np
import quimb.tensor as qtn
import cotengra as ctg

ADD = np.zeros((2,2,2)) # i1,i2,o
ADD[0,0,0] = ADD[0,1,1] = ADD[1,0,1] = 1.0
def delta(d):
    out = np.zeros((d,)*3)
    for i in range(d):
        out[i,i,i] = 1.0
    return out
d = 3
f = np.zeros((2,)+(d,)*3) # c,x1,x2,y
f[0,:,:,:] = np.ones((d,)*3)
f[1,:,:,:] = np.random.rand(d,d,d)
g = np.zeros((2,)+(d,)*3) # c,x1,x2,z
g[0,:,:,:] = np.ones((d,)*3)
g[1,:,:,:] = np.random.rand(d,d,d)
DEL = delta(d)
MUL = delta(2)
# addition
ls = []
ls.append(qtn.Tensor(data=f,inds=('i','x11','x21','y'),tags='f'))
ls.append(qtn.Tensor(data=g,inds=('j','x12','x22','z'),tags='g'))
ls.append(qtn.Tensor(data=ADD,inds=('i','j','k'),tags='+'))
ls.append(qtn.Tensor(data=DEL,inds=('x11','x12','x1'),tags='DEL'))
ls.append(qtn.Tensor(data=DEL,inds=('x21','x22','x2'),tags='DEL'))
tn = qtn.TensorNetwork(ls)
output_inds = ['k','x1','x2','y','z']
opt = ctg.HyperOptimizer()
out = tn.contract(output_inds=output_inds,optimize=opt)
x1 = np.random.randint(0,d)
x2 = np.random.randint(0,d)
y = np.random.randint(0,d)
z = np.random.randint(0,d)
print('check addition: ', f[1,x1,x2,y]+g[1,x1,x2,z]-out.data[1,x1,x2,y,z])
print('check addition: ', 1.0-out.data[0,x1,x2,y,z])
# multiplication
ls = []
ls.append(qtn.Tensor(data=f,inds=('i','x11','x21','y'),tags='f'))
ls.append(qtn.Tensor(data=g,inds=('j','x12','x22','z'),tags='g'))
ls.append(qtn.Tensor(data=MUL,inds=('i','j','k'),tags='*'))
ls.append(qtn.Tensor(data=DEL,inds=('x11','x12','x1'),tags='DEL'))
ls.append(qtn.Tensor(data=DEL,inds=('x21','x22','x2'),tags='DEL'))
tn = qtn.TensorNetwork(ls)
output_inds = ['k','x1','x2','y','z']
opt = ctg.HyperOptimizer()
out = tn.contract(output_inds=output_inds,optimize=opt)
x1 = np.random.randint(0,d)
x2 = np.random.randint(0,d)
y = np.random.randint(0,d)
z = np.random.randint(0,d)
print('check multiplication: ', f[1,x1,x2,y]*g[1,x1,x2,z]-out.data[1,x1,x2,y,z])
print('check multiplication: ', 1.0-out.data[0,x1,x2,y,z])

