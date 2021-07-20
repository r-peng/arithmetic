
#    tn = qtn.TensorNetwork([])
#    o = qtc.rand_uuid()
#    for i in range(len(xs)):
#        inds = 'x{},'.format(i),o
#        xi = qtn.Tensor(data=_tensor(xs[i]),inds=inds,tags={'x'})
#        tn.add_tensor(xi)
#        tn._outer_inds.add(inds[0])
#    tn._outer_inds.add(o)
#    tn._inner_inds.discard(o)
#    return tn
#def _train(xs):
#    # x1+...+xn
#    tn = qtn.TensorNetwork([])
#    indx = 'x0,',qtc.rand_uuid()
#    xi = qtn.Tensor(data=_tensor(xs[0]),inds=indx,tags={'x'})
#    indp = (indx[-1],)
#    tn.add_tensor(xi,tid=0,virtual=False)
#    tn._outer_inds.add(indx[0])
#    tn._inner_inds.add(indx[-1])
#    for i in range(1,len(xs)):
#        indx = 'x{},'.format(i),qtc.rand_uuid()
#        xi = qtn.Tensor(data=_tensor(xs[i]),inds=indx,tags={'x'})
#        indp = indp[-1],indx[-1],qtc.rand_uuid()
#        ai = qtn.Tensor(data=ADD,inds=indp,tags={'+'})
#        tn.add_tensor(xi)
#        tn.add_tensor(ai)
#        tn._outer_inds.add(indx[0])
#        tn._inner_inds.add(indx[-1])
#        if i==len(xs)-1:
#            tn._outer_inds.add(indp[-1])
#        else:
#            tn._inner_inds.add(indp[-1])
#    return tn
#
#def _mult_const(tn,a): #inplace
#    data = np.array([1.0,a])
#    o = list(tn._outer_inds._d.keys())[-1]
#    tn.add_tensor(qtn.Tensor(data=data,inds=(o,),tags={'c'}))
#    tn._outer_inds.add(o)
#    tn._inner_inds.discard(o)
