#!/usr/bin/env python3
import os
import numpy as np
import time

os.environ['OMP_NUM_THREADS'] = '1'

N = 2048

if __name__ == "__main__":


    a = np.random.rand(N, N).astype(np.float32)
    b = np.random.rand(N, N).astype(np.float32)


    flop = 2*N*N*N
    for i in range(100):
        tb = time.perf_counter()
        #C = np.matmul(a, b)
        C = a @ b                   # Doesn't matter wich one is used, both get numpy optimization
        te = time.perf_counter()

        s = te - tb

        print(f"GFLOPS {(flop / s) * 1e-9:.3f}")
