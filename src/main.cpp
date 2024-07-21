#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// extern void add(
//     float *a,
//     float *b,
//     float *c,
//     int n
// );

// void vector_addition() {

//     std::cout << "Program Start" << std::endl;

//     int n = 64;
//     size_t bytes = n * sizeof(float);

//     float *a = (float *)malloc(bytes);
//     float *b = (float *)malloc(bytes);
//     float *c = (float *)malloc(bytes);

//     for (int i = 0; i < n; i++) {

//         a[i] = 1.0;
//         b[i] = 1.0;
//     }

//     add(a, b, c, n);

//     for (int i = 0; i < n; i++) {

//         std::cout << c[i] << std::endl;
//     }

//     free(a);
//     free(b);
//     free(c);

//     std::cout << "Program End" << std::endl;
// }

// extern void transpose(
//     float *a,
//     float *b,
//     int N,
//     int M
// );

// void matrix_transpose() {

//     std::cout << "Program Start" << std::endl;

//     int N = 1 << 10;
//     int K = 1 << 10;
//     int M = 1 << 10;

//     float *a = (float *)malloc(N * K * sizeof(float));
//     float *b = (float *)malloc(K * M * sizeof(float));
//     float *c = (float *)malloc(N * M * sizeof(float));

//     matmul(a, b, c, N, K, M);

//     std::cout << "Program End" << std::endl;
// }

extern void matmul_cublas(
    uint M,
    uint K,
    uint N,
    const float *A,
    const float *B,
    float *C
);

extern void matmul(
    uint M,
    uint K,
    uint N,
    const float *A,
    const float *B,
    float *C
);

float random_float_range(float min, float max) {
    return min + ((float)rand() / (float)RAND_MAX) * (max - min);
}

void matrix_multiplication() {

    std::cout << "Program Start" << random_float_range(-1, 1) << std::endl;

    // uint M = 1 << 12;
    // uint K = 1 << 12;
    // uint N = 1 << 12;

    uint M = 1 << 7;
    uint K = 1 << 7;
    uint N = 1 << 7;

    std::cout << "N: " << N << std::endl;

    float *A = (float *)malloc(N * K * sizeof(float));
    float *B = (float *)malloc(K * M * sizeof(float));
    float *C_cublas = (float *)malloc(N * M * sizeof(float));
    float *C_custom = (float *)malloc(N * M * sizeof(float));

    for (uint i = 0; i < M * K; i++) {

        A[i] = random_float_range(-1, 1);
    }

    for (uint i = 0; i < K * N; i++) {

        B[i] = random_float_range(-1, 1);
    }

    // setting C to zeros
    memset(C_cublas, 0, N * M * sizeof(float));
    memset(C_custom, 0, N * M * sizeof(float));

    matmul_cublas(
        M,
        K,
        N,
        A,
        B,
        C_cublas
    );

    matmul(
        M,
        K,
        N,
        A,
        B,
        C_custom
    );

    for (uint i = 0; i < M * N; i++) {

        if (C_cublas[i] != C_custom[i]) {

            printf("%.10f %.10f\n", C_cublas[i], C_custom[i]);

            // std::cout << C_cublas[i] << " " << C_custom[i] << std::endl;
            // std::cout << C_cublas[i] - C_cublas[i] << std::endl;
        }
    }

    free(A);
    free(B);
    free(C_cublas);
    free(C_custom);

    std::cout << "Program End" << std::endl;
}

int main() {

    // vector_addition();
    // matrix_transpose();
    matrix_multiplication();

    return 0;
}
