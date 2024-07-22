#include <iostream>
#include <math.h>

#define SMEM_TILE_SIZE 32
#define REG_TILE_SIZE 2

#define REG_M REG_TILE_SIZE
#define REG_N REG_TILE_SIZE
#define REG_SIZE REG_M * REG_N

#define SMEM_M SMEM_TILE_SIZE * REG_M
#define SMEM_N SMEM_TILE_SIZE * REG_N
#define SMEM_SIZE SMEM_M * SMEM_N

__global__ void _matmul(
    uint M,
    uint K,
    uint N,
    const float *A,
    const float *B,
    float *C
) {

    __shared__ float A_smem[SMEM_SIZE];
    __shared__ float B_smem[SMEM_SIZE];

    float A_reg[REG_SIZE];
    float B_reg[REG_SIZE];
    float C_reg[REG_SIZE] = {0.0};

    uint threadX = threadIdx.x;
    uint threadY = threadIdx.y;

    uint globalX = blockIdx.x * blockDim.x + threadX;
    uint globalY = blockIdx.y * blockDim.y + threadY;

    uint numSmemTiles = (K / SMEM_N);
    uint numRegTiles = (SMEM_TILE_SIZE / REG_TILE_SIZE);

    for (uint smemTileIdx = 0; smemTileIdx < numSmemTiles; smemTileIdx++) {

        for (uint regTileY = 0; regTileY < REG_TILE_SIZE; regTileY++) {

            for (uint regTileX = 0; regTileX < REG_TILE_SIZE; regTileX++) {

                A_smem[(threadY * SMEM_N) + (regTileY * SMEM_N) + (threadX * REG_N) + regTileX] = A[(globalY * K) + (regTileY * K) + (smemTileIdx * SMEM_N) + (threadX * REG_N) + regTileX];
            }
        }

        for (uint regTileY = 0; regTileY < REG_TILE_SIZE; regTileY++) {

            for (uint regTileX = 0; regTileX < REG_TILE_SIZE; regTileX++) {

                B_smem[(threadY * SMEM_N) + (regTileY * SMEM_N) + (threadX * REG_N) + regTileX] = B[(smemTileIdx * SMEM_M * N) + (threadY * REG_M * N) + (globalX * REG_N) + regTileX];
            }
        }

        __syncthreads();

        for (uint regTileIdx = 0; regTileIdx < numRegTiles; regTileIdx++) {

            for (uint regTileY = 0; regTileY < REG_TILE_SIZE; regTileY++) {

                for (uint regTileX = 0; regTileX < REG_TILE_SIZE; regTileX++) {

                    A_reg[(regTileY * REG_TILE_SIZE) + regTileX] = A_smem[(threadY * SMEM_TILE_SIZE) + (threadX * REG_TILE_SIZE) + (regTileY * REG_TILE_SIZE) + regTileX];
                }
            }

            for (uint regTileY = 0; regTileY < REG_TILE_SIZE; regTileY++) {

                for (uint regTileX = 0; regTileX < REG_TILE_SIZE; regTileX++) {


                    B_reg[(regTileY * REG_TILE_SIZE) + regTileX] = B_smem[(threadY * SMEM_TILE_SIZE) + (threadX * REG_TILE_SIZE) + (regTileY * REG_TILE_SIZE) + regTileX];
                }
            }

            for (uint C_regY = 0; C_regY < REG_TILE_SIZE; C_regY++) {

                for (uint C_regX = 0; C_regX < REG_TILE_SIZE; C_regX++) {

                    for (uint dotIdx = 0; dotIdx < REG_TILE_SIZE; dotIdx++) {

                        C_reg[(C_regY * REG_TILE_SIZE) + C_regX] += A_reg[C_regY * REG_TILE_SIZE + dotIdx] * B_reg[C_regX + (dotIdx * REG_TILE_SIZE)];
                    }
                }
            }
        }

        __syncthreads();
    }

    // for (uint regTileIdx = 0; regTileIdx < numRegTiles; regTileIdx++) {

    //     C[???] = C_reg[???];
    // }
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

    dim3 gridDim(ceil(M / SMEM_TILE_SIZE), ceil(M / SMEM_TILE_SIZE), 1);
    dim3 blockDim(SMEM_TILE_SIZE, SMEM_TILE_SIZE, 1);

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
