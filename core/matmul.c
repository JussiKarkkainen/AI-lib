#include <stdio.h>
#include <stdint.h>
#include <time.h>


#define N 512ULL

void matmul();

uint64_t timer() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec*1000000000 + (uint64_t)start.tv_nsec;
}

float C[N][N];
float A[N][N];
float B[N][N]; 

void matmul() {
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

float A[N][N];
float B[N][N];

int main() {

    for (int u = 0; u <= 10; u++) {    
        uint64_t st = timer();
        matmul();
        uint64_t et = timer();


        double s = (et - st)*1e-9;
        double flop = (2*N*N*N)*1e-9;
        printf("GLOPS %f\n", (flop / s));

    }
    return 0;
}
