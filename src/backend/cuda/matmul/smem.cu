#include <iostream>
#include <math.h>

#define BLOCKSIZE 32

__global__ void _matmul(
    uint M,
    uint K,
    uint N,
    const float *A,
    const float *B,
    float *C
) {

    __shared__ float _A[BLOCKSIZE * BLOCKSIZE];
    __shared__ float _B[BLOCKSIZE * BLOCKSIZE];

    uint threadX = threadIdx.x;
    uint threadY = threadIdx.y;

    uint globalX = blockIdx.x * blockDim.x + threadX;
    uint globalY = blockIdx.y * blockDim.y + threadY;

    uint numTiles = (K / BLOCKSIZE);

    float value = 0;

    for (uint tileIdx = 0; tileIdx < numTiles; tileIdx++) {

        // printf("tileIdx: %d | globalX: %d | globalY: %d | threadX: %d | threadY: %d\n", tileIdx, globalX, globalY, threadX, threadY);

        // if (globalY == 0 && globalX == 0) {

        //     printf("A: %f | B: %f\n", A[(globalY * K) + (tileIdx * BLOCKSIZE) + threadX], B[globalX + (tileIdx * BLOCKSIZE * N) + (threadY * N)]);
        // }

        _A[threadY * BLOCKSIZE + threadX] = A[(globalY * K) + (tileIdx * BLOCKSIZE) + threadX];
        _B[threadY * BLOCKSIZE + threadX] = B[globalX + (tileIdx * BLOCKSIZE * N) + (threadY * N)];

        __syncthreads();

        // if (globalY < M && globalX < N) {

        for (int i = 0; i < BLOCKSIZE; i++) {

            value += _A[threadY * BLOCKSIZE + i] * _B[i * BLOCKSIZE + threadX];
        }

        __syncthreads();
        // }
    }

    C[globalY * N + globalX] += value;
}

void matmul(
    uint M, // rows of C / rows of A
    uint K, // columns of A / rows of B
    uint N, // columns of C / columns of B
    const float *A,
    const float *B,
    float *C
) {

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 gridDim(ceil(M / BLOCKSIZE), ceil(M / BLOCKSIZE), 1);
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE, 1);

    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc(&d_A, N * K * sizeof(float));
    cudaMalloc(&d_B, K * M * sizeof(float));
    cudaMalloc(&d_C, N * M * sizeof(float));

    cudaMemcpy(d_A, A, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * M * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start);

    _matmul<<<gridDim, blockDim>>>(
        M,
        K,
        N,
        d_A,
        d_B,
        d_C
    );

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "custom kernel execution time: " << milliseconds << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(C, d_C, N * M * sizeof(float), cudaMemcpyDeviceToHost);
}
