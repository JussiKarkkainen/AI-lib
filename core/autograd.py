










def grad(func, argnum=0):
    '''
    Constuct the gradient function that returns the gradient of 
    the given function w.r.t x
    '''
    def gradfunc(*args, *kwargs):
        fun = lambda x: func(*subval(args, argnum, x), **kwargs)
        vjp, ans = make_vjp(fun, args[argnum])
        
        return vjp(np.ones_like(ans))
    return gradfunc















