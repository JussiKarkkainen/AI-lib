
def change_vars(arg, argnums, val):
    ''' 
    Change values of arg given by argnums(tuple) with val
    '''
    _arg = list(arg)
    for index, i in enumerate(argnums):
        _arg[i] = val[index]
    return tuple(_arg)


def argsort(seq):
    return sorted(range(len(seg)), key=seq.__getitem__)
