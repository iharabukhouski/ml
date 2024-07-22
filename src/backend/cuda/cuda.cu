#include <iostream>

void info() {

    cudaDeviceProp props;
    int deviceId = 0;

    cudaError_t error = cudaGetDeviceProperties(&props, deviceId);

    printf("\n");
    printf("CUDA Device Props\n");
    printf("Name: %s\n", props.name);
    printf("Compute Capability: %d.%d\n", props.major, props.minor);
    printf("Global Memory (bytes): %zu\n", props.totalGlobalMem);
    printf("Max Threads Per Block: %d\n", props.maxThreadsPerBlock);
    printf("Warp Size: %d\n", props.warpSize);
    printf("Regs Per Block: %d\n", props.regsPerBlock);
    // printf("Regs Per SM: %d\n", props.regsPerMultiProcessor);
    printf("Num SMs: %d\n", props.multiProcessorCount);
    printf("Shared Mem Per Block: %zu\n", props.sharedMemPerBlock);
    printf("\n");
}
