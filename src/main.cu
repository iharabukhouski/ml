#include "stdio.h"

__global__ void my_simple_kernel(int *a) {

    printf("Start: %d\n");
    printf("X: %d\n", threadIdx.x);
    printf("Y: %d\n", threadIdx.y);
    printf("\n");

    *a = 3;
}


int main() {

    printf("Hi\n");

    int a;
    int* d_a;

    cudaMalloc(&d_a, sizeof(int));

    dim3 gridDim(1, 1, 1);
    dim3 blockDim(2, 1, 1);

    my_simple_kernel<<<gridDim, blockDim>>>(d_a);

    // cudaMemcpy(y, d_y, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&a, d_a, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result: %d\n", a);

    return 0;
}
