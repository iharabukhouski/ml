void matmul_cpu(
    const uint M,
    const uint K,
    const uint N,
    const float *A,
    const float *B,
    float *C
) {

    for (uint m = 0; m < M; m++) {

        for (uint n = 0; n < N; n++) {

            for (uint k = 0; k < K; k++) {

                C[m * N + n] += A[m * K + k] * B[k * N + n];
            }
        }
    }
}
