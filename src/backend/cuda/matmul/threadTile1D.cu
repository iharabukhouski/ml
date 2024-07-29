#include "stdio.h"

// tile that uses: shared memory (aka l2 cache) & multiple threads
// #define BM 2
// #define BK 2
// #define BN 2

// tile that uses: registers (aka registerfile) & single thread
// #define TM 2
// #define TN 2

const uint BM = 2;
const uint BK = 2;
const uint BN = 2;

const uint TM = 2;

__global__ void _matmul(
    uint M,
    uint K,
    uint N,
    const float *A,
    const float *B,
    float *C
) {

    uint row_A_smem = threadIdx.x / BK;
    uint col_A_smem = threadIdx.x % BK;
    uint row_B_smem = threadIdx.x / BN;
    uint col_B_smem = threadIdx.x % BN;
    uint row_C = threadIdx.x / BN;
    uint col_C = threadIdx.x % BN;

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    __shared__ float A_smem[BM * BK];
    __shared__ float B_smem[BK * BN];

    float threadResults[TM] = {0.0};

    for (uint blockTileIdx = 0; blockTileIdx < K; blockTileIdx += BK) {

        A_smem[row_A_smem * BK + col_A_smem] = A[row_A_smem * BK + col_A_smem];
        B_smem[row_B_smem * BN + col_B_smem] = B[row_B_smem * BN + col_B_smem];

        __syncthreads();

        A += BK;
        B += BK * BN;

        for (uint threadResultIdx = 0; threadResultIdx < TM; threadResultIdx++) {

            for (uint dotIdx = 0; dotIdx < BK; dotIdx++) {

                threadResults[threadResultIdx] += A_smem[(row_C * TM + threadResultIdx) * BK + dotIdx] * B_smem[dotIdx * BN + col_C];
            }
        }

        __syncthreads();
    }

    for (uint threadResultIdx = 0; threadResultIdx < TM; threadResultIdx++) {

        C[(row_C * TM + threadResultIdx) * N + col_C] = threadResults[threadResultIdx]; 
    }
}

void matmul(
    uint M,
    uint K,
    uint N,
    const float *A,
    const float *B,
    float *C
) {

    float *d_A;
    float *d_B;
    float *d_C;

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridDim(N / BN, M / BM, 1);
    dim3 blockDim(BN * BM / TM, 1, 1);

    _matmul<<<gridDim, blockDim>>>(
        M,
        K,
        N,
        d_A,
        d_B,
        d_C
    );

    cudaDeviceSynchronize();
}
