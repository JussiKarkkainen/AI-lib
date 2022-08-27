import numpy as np

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

def get_im2col_indices(x_shape, field_height, field_width, padding, stride):
    N, C, H, W = x_shape
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1
    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), int(out_height))
    i = (i0.reshape(-1, 1) + i1.reshape(1, -1)).astype(np.int64)
    j = (j0.reshape(-1, 1) + j1.reshape(1, -1)).astype(np.int64)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding, stride):
    p = padding
    x_pad = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    cols = x_pad[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape((C * field_height * field_width, -1, N))
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]




