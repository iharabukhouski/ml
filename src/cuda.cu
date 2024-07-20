#include <iostream>

// grid -> block -> thread
// warp (SIMT) - lowest schedulable unit


// gridDim / blockIdx.x / blockDim.x / threadIdx.x

// grids of threads
// grid can be multi dimentional
// grid -> thread block -> thread
// warp is 32 threads
// <<blocks, threads>>

__global__ void _transpose(
    // clock_t *time
) {

    // if (threadIdx.x == 0) {

    //     time[blockIdx.x] = clock();
    // }

    // if (treadIdx.x == 0) {

    //     time[blockIdx.x + gridDim.x] = clock();
    // }
}

void transpose() {


}

# define BLOCK_SIZE 32

template<
    const uint TILESIZE
>
__global__ void _matmul2(
    uint M, // rows of C / rows of A
    uint K, // columns of A / rows of B
    uint N, // columns of C / columns of B
    const float *A,
    const float *B,
    float *C
) {

    __shared__ float _A[TILESIZE * TILESIZE];
    __shared__ float _B[TILESIZE * TILESIZE];

    const uint tileY = blockIdx.y;
    const uint tileX = blockIdx.x;

    const uint threadY = threadIdx.y;
    const uint threadX = threadIdx.x;

    A += tileY * TILESIZE * K;
    B += tileX * TILESIZE;
    C += tileY * TILESIZE * N + tileX * TILESIZE;

    float c_m_n = 0;

    for (uint tileIdx = 0; tileIdx < K; tileIdx += TILESIZE) {

        _A[threadY * TILESIZE + threadX] = A[threadY * K + threadX];
        _B[threadY * TILESIZE + threadX] = B[threadY * N + threadX];

        __syncthreads();

        A += TILESIZE;
        B += TILESIZE * N;

        for (uint i = 0; i < alkdj; i++) {

            for (uint elementIdx = 0; elementIdx < TILESIZE; elementIdx++) {

                c_m_n += _A[threadY * TILESIZE + elementIdx] * _B[threadX + TILESIZE * elementIdx];
            }

        }

        __syncthreads();
    }

    C[threadY * N + threadX] = c_m_n;
}

template<
    const uint BM, // block tile
    const uint BK, // block tile
    const uint BN, // block tile
    const uint TM, // thread tile
    const uint TN, // thread tile
>
__global__ void _matmul3(
    uint M, // rows of C / rows of A
    uint K, // columns of A / rows of B
    uint N, // columns of C / columns of B
    const float *A,
    const float *B,
    float *C
) {

    __shared__ float _A[BM * BK];
    __shared__ float _B[BK * BN];

    blockIdx.x;
    blockIdx.y;

    threadIdx.x;
    threadIdx.y;

    for (uint tm = 0; tm < TM; tm++) { // iterate over rows

        for (uint tn = 0; tn < TN; tn++) { // iterate over columns

            _A[threadIdx.x * TN + tn + threadIdx.y * tm * TM]
        }
    }

}

#define CEIL_DIV(M, N) ((M + N - 1) / N)

void matmul(
    uint M, // rows of C / rows of A
    uint K, // columns of A / rows of B
    uint N, // columns of C / columns of B
    const float *A,
    const float *B,
    float *C
) {

    dim3 gridDim(CEIL_DIV(N, BLOCK_SIZE), CEIL_DIV(M, BLOCK_SIZE), 1);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc(&d_A, N * K * sizeof(float));
    cudaMalloc(&d_B, K * M * sizeof(float));
    cudaMalloc(&d_C, N * M * sizeof(float));

    cudaMemcpy(d_A, A, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * M * sizeof(float), cudaMemcpyHostToDevice);

    _matmul<<<gridDim, blockDim>>>(
        M,
        K,
        N,
        d_A,
        d_B,
        d_C
    );

    cudaMemcpy(C, d_C, N * M * sizeof(float), cudaMemcpyDeviceToHost)
}

__global__ void _add(
    float *a,
    float *b,
    float *c,
    int n
) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {

        c[i] = a[i] + b[i];
    }
}

void add(
    float *a,
    float *b,
    float *c,
    int n
) {

    size_t bytes = n * sizeof(float);

    float *d_c;
    float *d_a;
    float *d_b;
    // dim3 blocks(1, 1);
    // dim3 threads(1, 1);

    cudaMalloc(&d_c, bytes);
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);

    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

    _add<<<1, 128>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_c);
    cudaFree(d_a);
    cudaFree(d_b);

    std::cout << std::endl;
}
