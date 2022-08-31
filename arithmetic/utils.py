import numpy as np
import quimb.tensor as qtn
import pickle,time
ADD = np.zeros((2,)*3)
ADD[0,1,1] = ADD[1,0,1] = ADD[0,0,0] = 1.
CP2 = np.zeros((2,)*3)
CP2[0,0,0] = CP2[1,1,1] = 1.
#####################################################################################
# MPI STUFF
#####################################################################################
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
def parallelized_looped_function(func, iterate_over, args, kwargs):
    """
    When a function must be called many times for a set of parameters
    then this implements a parallelized loop controlled by the rank 0 process.
    
    Args:
        func: Function
            The function that will be called. 
        iterate_over: iterable
            The argument of the function that will be iterated over.
        args: list
            A list of arguments to be supplied to the function
        kwargs: dict
            A dictrionary of arguments to be supplied to the function
            at each call

    Returns:
        results: list
            This is a list of the results of each function call stored in 
            a list with the same ordering as was supplied in 'iterate_over'
    """
    # Figure out which items are done by which worker
    min_per_worker = len(iterate_over) // SIZE

    per_worker = [min_per_worker for _ in range(SIZE)]
    for i in range(len(iterate_over) - min_per_worker * SIZE):
        per_worker[SIZE-1-i] += 1

    randomly_permuted_tasks = np.random.permutation(len(iterate_over))
    worker_ranges = []
    for worker in range(SIZE):
        start = sum(per_worker[:worker])
        end = sum(per_worker[:worker+1])
        tasks = [randomly_permuted_tasks[ind] for ind in range(start, end)]
        worker_ranges.append(tasks)

    # Container for all the results
    worker_results = [None for _ in range(SIZE)]

    # Loop over all the processes (backwards so zero starts last
    for worker in reversed(range(SIZE)):

        # Collect all info needed for workers
        worker_iterate_over = [iterate_over[i] for i in worker_ranges[worker]]
        worker_info = [func, worker_iterate_over, args, kwargs]

        # Send to worker
        if worker != 0:
            COMM.send(worker_info, dest=worker)

        # Do task with this worker
        else:
            worker_results[0] = [None for _ in worker_ranges[worker]]
            for func_call in range(len(worker_iterate_over)):
                result = func(worker_iterate_over[func_call], 
                              *args, **kwargs)
                worker_results[0][func_call] = result

    # Collect all the results
    for worker in range(1, SIZE):
        worker_results[worker] = COMM.recv(source=worker)

    results = [None for _ in range(len(iterate_over))]
    for worker in range(SIZE):
        worker_ind = 0
        for i in worker_ranges[worker]:
            results[i] = worker_results[worker][worker_ind]
            worker_ind += 1

    # Return the results
    return results

def worker_execution():
    """
    All but the rank 0 process should initially be called
    with this function. It is an infinite loop that continuously 
    checks if an assignment has been given to this process. 
    Once an assignment is recieved, it is executed and sends
    the results back to the rank 0 process. 
    """
    # Create an infinite loop
    while True:

        # Loop to see if this process has a message
        # (helps keep processor usage low so other workers
        #  can use this process until it is needed)
        while not COMM.Iprobe(source=0):
            time.sleep(0.01)

        # Recieve the assignments from RANK 0
        assignment = COMM.recv()

        # End execution if received message 'finished'
        if assignment == 'finished': 
            break

        # Otherwise, call function
        function = assignment[0]
        iterate_over = assignment[1]
        args = assignment[2]
        kwargs = assignment[3]
        results = [None for _ in range(len(iterate_over))]
        for func_call in range(len(iterate_over)):
            results[func_call] = function(iterate_over[func_call], 
                                          *args, **kwargs)

        # Send the results back to the rank 0 process
        COMM.send(results, dest=0)
#####################################################################################
# READ & WRITE 
#####################################################################################
def load_tn_from_disc(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    tn = qtn.TensorNetwork([])
    for tid,ten in data['tensors'].items():
        T = qtn.Tensor(ten.data, inds=ten.inds, tags=ten.tags)
        tn.add_tensor(T,tid=tid,virtual=True)
    extra_props = dict()
    for name,prop in data['tn_info'].items():
        extra_props[name[1:]] = prop
    tn.exponent = data['exponent']
    tn = tn.view_as_(data['class'], **extra_props)
    return tn
def write_tn_to_disc(tn,fname):
    data = dict()
    data['class'] = type(tn)
    data['tensors'] = dict()
    for tid,T in tn.tensor_map.items():
        data['tensors'][tid] = T 
    data['tn_info'] = dict()
    for e in tn._EXTRA_PROPS:
        data['tn_info'][e] = getattr(tn, e)
    data['exponent'] = tn.exponent
    with open(fname, 'wb') as f:
        pickle.dump(data, f)
    return 

def scale(tn):
    for tid in tn.tensor_map:
        T = tn.tensor_map[tid]
        fac = np.amax(np.absolute(T.data))
        T.modify(data=T.data/fac)
        tn.exponent += np.log10(fac)
    return tn

#def get_cheb_coeff(fxn,order,a=-1.,b=1.):
#    N = order + 1
#    c = []
#    theta = np.array([np.pi*(k-0.5)/N for k in range(1,N+1)])
#    x = np.cos(theta)*(b-a)/2.+(b+a)/2.
#    for j in range(order+1):
#        v1 = np.array([fxn(xk) for xk in x])
#        v2 = np.array([np.cos(j*thetak) for thetak in theta])
#        c.append(np.dot(v1,v2)*2./N)
#    coeff = np.polynomial.chebyshev.cheb2poly(c)
#    coeff[0] -= 0.5*c[0]
#
#    A,B = 2./(b-a),-(b+a)/(b-a)
#    c = np.zeros_like(coeff)
#    fac = [1]
#    for i in range(1,order+1):
#        fac.append(fac[-1]*i)
#    for i in range(order+1):
#        for j in range(i+1):
#            c[j] += coeff[i]*A**j*B**(i-j)*fac[i]/(fac[i-j]*fac[j])
#    return c
#def compress1D(tn,tag,maxiter=10,final='left',shift=0,iprint=0,**compress_opts):
#    L = tn.num_tensors
#    max_bond = tn.max_bond()
#    lrange = range(shift,L-1+shift)
#    rrange = range(L-1+shift,shift,-1)
#    if iprint>0:
#        print('init max_bond',max_bond)
#    def canonize_from_left():
#        if iprint>1:
#            print('canonizing from left...')
#        for i in lrange:
#            if iprint>2:
#                print(f'canonizing between {tag}{i},{i+1}...')
#            tn.canonize_between(f'{tag}{i}',f'{tag}{i+1}',absorb='right')
#    def canonize_from_right():
#        if iprint>1:
#            print('canonizing from right...')
#        for i in rrange:
#            if iprint>2:
#                print(f'canonizing between {tag}{i},{i-1}...')
#            tn.canonize_between(f'{tag}{i-1}',f'{tag}{i}',absorb='left')
#    def compress_from_left():
#        if iprint>1:
#            print('compressing from left...')
#        for i in lrange:
#            if iprint>2:
#                print(f'compressing between {tag}{i},{i+1}...')
#            tn.compress_between(f'{tag}{i}',f'{tag}{i+1}',absorb='right',**compress_opts)
#    def compress_from_right():
#        if iprint>1:
#            print('compressing from right...')
#        for i in rrange:
#            if iprint>2:
#                print(f'compressing between {tag}{i},{i-1}...')
#            tn.compress_between(f'{tag}{i-1}',f'{tag}{i}',absorb='left',**compress_opts)
#    if final=='left':
#        canonize_from_left()
#        def sweep():
#            compress_from_right()
#            compress_from_left()
#    elif final=='right':
#        canonize_from_right()
#        def sweep():
#            compress_from_left()
#            compress_from_right()
#    else:
#        raise NotImplementedError(f'{final} canonical form not implemented!')
#    for i in range(maxiter):
#        sweep()
#        max_bond_new = tn.max_bond()
#        if iprint>0:
#            print(f'iter={i},max_bond={max_bond_new}')
#        if max_bond==max_bond_new:
#            break
#        max_bond = max_bond_new
#    return tn
