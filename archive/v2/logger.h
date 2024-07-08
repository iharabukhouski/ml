#pragma once

#include "tensor.h"

// // #include "ml.h"

// void print(const char* a);

void print2(int a);

template<
    int N,
    int M,
    typename T
>
void print(
    const Tensor<T, N, M>& a
) {

    // std::cout << std::endl << "raw = [ ";

    // for (int i = 0; i < N * M; i++) {

    //     std::cout << a.data->at(i) << ", ";
    // }

    // std::cout << "]" << std::endl;

    std::cout << std::endl << "data = [" << std::endl;

    for (int n = 0; n < N; n++) {

        for (int m = 0; m < M; m++) {

            int stride = a.data->stride;

            int i = a.data->contiguous == true ? (m * stride) + n * M : (m * stride) + n;

            std::cout << "  " << a.data->at(i) << ", ";
        }

        std::cout << std::endl;
    }

    std::cout << "]" << std::endl;

    std::cout << std::endl << "grad = [" << std::endl;

    for (int n = 0; n < N; n++) {

        for (int m = 0; m < M; m++) {

            std::cout << "  " << a.grad->at(n * M + m) << ", ";
        }

        std::cout << std::endl;
    }

    std::cout << "]" << std::endl;
}

void println() {

    std::cout << std::endl;
}
