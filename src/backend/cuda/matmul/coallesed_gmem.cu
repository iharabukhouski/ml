#include <iostream>

const uint BLOCKSIZE = 32;

__global__ void _matmul(
    uint M,
    uint K,
    uint N,
    const float *A,
    const float *B,
    float *C
) {

    const uint n = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const uint m = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    float value = 0;

    for (uint k = 0; k < K; k++) {

        value += A[m * K + k] * B[k * N + n];
    }

    C[m * N + n] = value;
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

    dim3 gridDim(M / BLOCKSIZE, N / BLOCKSIZE, 1);
    dim3 blockDim(BLOCKSIZE * BLOCKSIZE, 1, 1);

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

    std::cout << "kernel execution time: " << milliseconds << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(C, d_C, N * M * sizeof(float), cudaMemcpyDeviceToHost);
}
