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

__global__ void _matmul(
    float *a,
    float *b,
    float *c,
    int N,
    int K,
    int M
) {

    int threads = 16;
    int tile_size = threads;

    __shared__ int A[threads * threads * sizeof(float)];
    __shared__ int B[threads * threads * sizeof(float)];

    int x, m = blockIdx.x * blockDim.x + threadIdx.x;
    int y, n = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < (N / tile_size); i++) {

        if ((n < N) && (m < M)) {

            int value = 0;

            for (int k = 0; k < K; k++) {

                value += a[n * K + k] * b[];
            }

            c[n * M + m] = value;
        }
    }
}

void matmul(
    float *a,
    float *b,
    float *c,
    int N,
    int K,
    int M
) {

    dim3 numBlocks(64, 64);
    dim3 threadsPerBlock(16, 16);

    float* d_a;
    float* d_b;
    float* d_c;

    cudaMalloc(&d_a, N * K * sizeof(float));
    cudaMalloc(&d_b, K * M * sizeof(float));
    cudaMalloc(&d_c, N * M * sizeof(float));

    cudaMemcpy(d_a, a, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, K * M * sizeof(float), cudaMemcpyHostToDevice);

    _matmul<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N, K, M);

    cudaMemcpy(c, d_c, N * M * sizeof(float), cudaMemcpyDeviceToHost)
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
