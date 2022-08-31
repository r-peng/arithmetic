import numpy as np
import arithmetic.pp.gauss as gauss 
import scipy,functools
import quimb.tensor as qtn
import matplotlib.pyplot as plt

xmax = 2.
beta = 1.
density_fxn = functools.partial(gauss.lennard_jones_density,beta=beta)
#x = np.linspace(0.,xmax,30)
#y = np.linspace(0.,xmax,30)
#X,Y = np.meshgrid(x,y)
#def plot(Z,fname):
#    fig = plt.figure()
#    ax = plt.axes(projection='3d')
#    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
#                    cmap='viridis', edgecolor='none')
#    ax.set_xlabel('x')
#    ax.set_ylabel('y')
#    ax.set_zlabel('z')
#    fig.savefig(fname,dpi=300)
#    plt.close()

#Z = np.zeros((30,30))
#for i in range(30):
#    for j in range(30):
#        Z[i,j] = density_fxn(np.array([x[i],y[j]]))
#plot(Z,'LJ.png')
#exit()

ng = 30
ranges = (0.,xmax),(0.,xmax)
points,weights = gauss.get_quad(ranges,ng)
density = np.array([density_fxn(point) for point in points])
I = np.dot(weights,density)
print(f'ng={ng},I={I}')

ng = 10
nbasis = 3
ndim = 2
centers,variances,coeffs = gauss.initialize(xmax,ng,nbasis,ndim,density_fxn)
coeffs /= I
print('sum coeffs=',np.sum(coeffs))
print('coeffs=',coeffs)

density /= I
basis,_,_ = gauss.batched_basis_fxns(centers,variances,points)
data = gauss.optimize_coeffs(basis,density,weights,affine=True)
for key,(coeffs,loss) in data.items():
    print(key,loss,gauss.loss(coeffs,basis,density,weights),coeffs)
exit()

SQRT_PI = np.sqrt(np.pi)
def gauss(r,b,c):
    vec = r-b
    return (c/SQRT_PI)**ndim*np.exp(-c**2*np.dot(vec,vec)) 
Z = np.zeros((len(coeffs),30,30))
for i in range(len(coeffs)):
    for i_ in range(30):
        for j_ in range(30):
            Z[i,i_,j_] = gauss(np.array([x[i_],y[j_]]),centers[i,:],variances[i])
    #plot(Z[i,:,:],f'{i}.png')
    #print(f'i={i},coeff={coeffs[i]}')
Z = np.einsum('i,ixy->xy',coeffs/I,Z)
plot(Z,'SumGauss.png')
