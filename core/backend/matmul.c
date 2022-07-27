// gcc -O3 -ffast-math matmul.c && ./a.out

#include <stdio.h>
#include <stdint.h>
#include <time.h>

#define N 1024ULL           // Size of matrix
#define bs 16 
#define num_blocks (N / bs)  // number of blocks

void matmul(); 

uint64_t timer() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec*1000000000 + (uint64_t)start.tv_nsec;
}

float C[N][N] __attribute__((aligned (32)));
float A[N][N] __attribute__((aligned (32)));
float B[N][N] __attribute__((aligned (32))); 

void matmul() {

    for (int ii = 0; ii < N; ii+=bs) {
        for (int jj = 0; jj < N; jj+=bs) {
            for (int kk = 0; kk < N; kk+=bs) {
                for (int i = 0; i < bs; i++) {
                    for (int j = 0; j < bs; j++) {
                        for (int k = 0; k < bs; k++) {
                            C[ii+i][jj+j] += A[ii+i][kk+k] * B[kk+k][jj+j];
                
                        }
                    }
                }
            }
        }
    }
}

int main() {
    printf("starting\n");
/*
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            A[i][j] = 1;
            B[i][j] = 2;
        }
    } 
*/  
    for (int u = 0; u <= 10; u++) {    
        uint64_t st = timer();
        matmul(A, B, C);
        uint64_t et = timer();

        double s = (et - st)*1e-9;
        double flop = (2*N*N*N)*1e-9;
        printf("GFLOPS %f\n", (flop / s));

    }
    
    return 0;
}
