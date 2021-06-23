import numpy as np
import math
import rus, oaa
ZERO = np.array([1.0,0.0])
xs = [np.random.rand() for i in range(2)]
w = np.random.rand()
def _num_oaa(L,xs,gamma=0.5):
    fac = math.sqrt(1-gamma**2)
    phi = []
    for j in range(L):
        tmp = 2*math.pi*(j+1)/(2*L+1)
        tmp = math.tan(tmp)*fac
        # acot=pi/2-atan
        tmp = math.pi/2-math.atan(tmp)
        phi.append(-2*tmp)
    amps = []
    for x in xs:
        A = np.array([[x,math.sqrt(1-x**2)],
                   [math.sqrt(1-x**2),-x]])
        AL = np.array(A,dtype=complex)
        for j in range(L):
            Sl = np.array([np.exp(1j*phi[j]),1],dtype=complex)
            Sl = np.diag(Sl)
            j_ = L-1-j
            Sr = np.array([np.exp(1j*phi[j_]),1],dtype=complex)
            Sr = np.diag(Sr)
            G = np.linalg.multi_dot([A,Sl,A,Sr])
            AL = np.dot(-G,AL)
        amps.append(AL[0,0])
    return amps
R1 = rus._Rys(xs)
test = 'L'
if test == 'G':
    a = np.random.rand()
    b = np.random.rand()
    Sa = np.array([[np.exp(1j*a),0.0],[0.0,1.0]],dtype=complex)
    Sb = np.array([[np.exp(1j*b),0.0],[0.0,1.0]],dtype=complex)
    print('########### check G for GB #############')
    R = rus._gb(R1,contr_ancilla=False)
    G1 = oaa._G1(b,a,R) 
    G1 = np.einsum('...ijkl,i,j->...kl',G1,ZERO,ZERO)
    U = np.einsum('...ijkl,i,j->...kl',R,ZERO,ZERO)
    for i in range(len(xs)):
        amp = np.linalg.norm(U[i,:,0]) # amp=\sqrt{\lambda_0}
        amp_ = math.sqrt(1-amp**2) # amp_=\sqrt{\lambda_i}
        A = np.array([[amp,amp_],[amp_,-amp]])
        G = - np.linalg.multi_dot([A,Sa,A,Sb])
        print(G[0,0])
        print(G1[i,...])
    # PAR
    print('######### check G for PAR ########')
    R = rus._par(R1,w,contr_ancilla=False)
    G1 = oaa._G2(b,a,R) 
    G1 = np.einsum('...ijklmn,i,j,k,l->...mn',G1,ZERO,ZERO,ZERO,ZERO)
    U = np.einsum('...ijklmn,i,j,k,l->...mn',R,ZERO,ZERO,ZERO,ZERO)
    for i in range(len(xs)):
        amp = np.linalg.norm(U[i,:,0]) # amp=\sqrt{\lambda_0}
        amp_ = math.sqrt(1-amp**2) # amp_=\sqrt{\lambda_i}
        A = np.array([[amp,amp_],[amp_,-amp]])
        G = - np.linalg.multi_dot([A,Sa,A,Sb])
        print(G[0,0])
        print(G1[i,...])
if test == 'L':
    print('####### check L convergence for GB ###########')
    R = rus._gb(R1,contr_ancilla=False)
    U = np.einsum('...ijkl,i,j->...kl',R,ZERO,ZERO)
    normed = []
    amps = []
    err_angle = 0
    err_amp = 0
    for i in range(len(xs)):
        amps.append(np.linalg.norm(U[i,:,0]))
        normed.append(U[i,...]/amps[-1])
    print(amps)
    for L in [0,4,10]:
        print('######## L={} #############'.format(L))
        amp1 = _num_oaa(L,amps)
        print(amp1)
        angles = oaa._angles(L)
        GA = oaa._normalize1(R,angles)
        GA = np.einsum('...ijkl,i,j->...kl',GA,ZERO,ZERO)
        for i in range(len(xs)):
            tan = GA[i,1,0]/GA[i,0,0]
            err_angle += abs(math.atan(tan.real)-math.atan(U[i,1,0]/U[i,0,0]))
            err_angle += abs(tan.imag)
            print('normsq', np.linalg.norm(GA[i,:,0])**2)
            amp2 = np.divide(GA[i,...],normed[i])
            err_amp += np.linalg.norm(amp1[i]*np.ones((2,)*2)-amp2)
    print('angle err', err_angle)
    print('amp err', err_amp)
    print('####### check L convergence for PAR ###########')
    R = rus._par(R1,w,contr_ancilla=False)
    U = np.einsum('...ijklmn,i,j,k,l->...mn',R,ZERO,ZERO,ZERO,ZERO)
    normed = []
    amps = []
    err_angle = 0
    err_amp = 0
    for i in range(len(xs)):
        amps.append(np.linalg.norm(U[i,:,0]))
        normed.append(U[i,...]/amps[-1])
    print(amps)
    for L in [0,4,10]:
        print('######## L={} #############'.format(L))
        amp1 = _num_oaa(L,amps)
        print(amp1)
        angles = oaa._angles(L)
        GA = oaa._normalize2(R,angles)
        GA = np.einsum('...ijklmn,i,j,k,l->...mn',GA,ZERO,ZERO,ZERO,ZERO)
        for i in range(len(xs)):
            tan = GA[i,1,0]/GA[i,0,0]
            err_angle += abs(math.atan(tan.real)-math.atan(U[i,1,0]/U[i,0,0]))
            err_angle += abs(tan.imag)
            print('normsq', np.linalg.norm(GA[i,:,0])**2)
            amp2 = np.divide(GA[i,...],normed[i])
            err_amp += np.linalg.norm(amp1[i]*np.ones((2,)*2)-amp2)
    print('angle err', err_angle)
    print('amp err', err_amp)
