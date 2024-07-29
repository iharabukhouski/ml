#include <stdio.h>
#include <stdlib.h>

extern void matmul(
    uint M,
    uint K,
    uint N,
    const float *A,
    const float *B,
    float *C
);

void ones(
    uint M,
    uint N,
    float *A
) {

    for (uint i = 0; i < M * N; i++) {

        A[i] = 1;
    }
}

void zeros(
    int M,
    int N,
    float *A
) {

    for (int i = 0; i < M * N; i++) {

        A[i] = 0;
    }
}

void range(
    uint M,
    uint N,
    float *A
) {

    for (uint i = 0; i < M * N; i++) {

        A[i] = i + 2;
    }
}

int main() {

    printf("Program Start\n");

    uint M = 1 << 2;
    uint K = 1 << 2;
    uint N = 1 << 2;

    printf("N: %d\n", N);

    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * N * sizeof(float));
    float *C = (float *)malloc(M * N * sizeof(float));

    range(M, K, A);
    range(K, N, B);
    zeros(M, N, C);

    matmul(
        M,
        K,
        N,
        A,
        B,
        C
    );

    printf("Program End\n");

    return 0;
}
