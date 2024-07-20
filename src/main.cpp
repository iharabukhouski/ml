#include <iostream>

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

extern void matmul(
    uint M,
    uint K,
    uint N,
    const float *A,
    const float *B,
    float *C
);

void matrix_multiplication() {

    std::cout << "Program Start" << std::endl;

    uint N = 1 << 12;
    uint K = 1 << 12;
    uint M = 1 << 12;

    std::cout << "N:" << N << std::endl;

    const float *A = (const float *)malloc(N * K * sizeof(float));
    const float *B = (const float *)malloc(K * M * sizeof(float));
    float *C = (float *)malloc(N * M * sizeof(float));

    matmul(
        M,
        K,
        N,
        A,
        B,
        C
    );

    free((float *)A);
    free((float *)B);
    free(C);

    std::cout << "Program End" << std::endl;
}

int main() {

    // vector_addition();
    // matrix_transpose();
    matrix_multiplication();

    return 0;
}
