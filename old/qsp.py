import pyqsp, math
from pyqsp.angle_sequence import QuantumSignalProcessingPhases as QSPP
operator = 'Wx'
tol = 1e-6
print('######### cos ###########')
coeff = [1.0,0,-1.0/2,0,1.0/24,0,-1.0/720,0,1.0/40320,0,-1.0/3628800]
angles = QSPP(coeff,signal_operator=operator,tolerence=tol)
print(angles)
print('######### sin ###########')
coeff = [0,1.0,0,-1.0/6,0,1.0/120,0,-1.0/5040,0,1.0/362880]
angles = QSPP(coeff,signal_operator=operator)
print(angles)
