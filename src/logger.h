#pragma once

#include "tensor.h"

// // #include "ml.h"

// void print(const char* a);

// void print(int a);

// // void print(Tensor* a);

template<typename T, int N, int M>
void print(Tensor<T, N, M> a) {

    for (int n = 0; n < N; n++) {

        for (int m = 0; m < M; m++) {

            std::cout << a.buffer->at(n * M + m) << ", ";
        }

        std::cout << std::endl;
    }
}
