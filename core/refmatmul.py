#!/usr/bin/env python3
import os
import numpy as np
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'

N = 2048

if __name__ == "__main__":


    a = np.random.randn(N, N).astype(np.float32)
    b = np.random.randn(N, N).astype(np.float32)


    flop = 2*N*N*N
    for i in range(100):
        tb = time.perf_counter()
        #C = np.matmul(a, b)
        C = a @ b
        te = time.perf_counter()

        s = te - tb

        print(f"GFLOPS {(flop / s) * 1e-9:.3f}")
