#pragma once

#include <iostream>
#include <algorithm>
#include <array>
#include <random>
#include <ctime>
#include "logger.h"
#include "tensor.h"

template<typename T, int n_rows, int n_cols>
Tensor<T, n_rows, n_cols> randn() {

    std::array<T, n_rows * n_cols>* buffer = new std::array<T, n_rows * n_cols>();

    std::default_random_engine generator(std::time(0));
    std::normal_distribution<T> distribution(0, 1);

    for (int i = 0; i < n_rows * n_cols; i++) {

        buffer->at(i) = distribution(generator);
    }

    return Tensor<T, n_rows, n_cols>(buffer);
};

template<typename T, int N, int M>
Tensor<T, N, M> relu(
    Tensor<T, N, M>& a
) {

    std::array<T, N * M>* buffer = new std::array<T, N * M>();

    for (int i = 0; i < a.buffer->size(); i++) {

        std::cout << a.buffer->at(i) << std::endl;

        // TODO: without this fails with "error: no matching function for call to 'max'"
        T zero = 0;

        buffer->at(i) = std::max(zero, a.buffer->at(i));
    }

    return Tensor<T, N, M>(buffer);
};

template<typename T, int N, int K, int M>
Tensor<T, N, M> matmul(
    Tensor<T, N, K>& a,
    Tensor<T, K, M>& b
) {

    std::array<T, N * M>* buffer = new std::array<T, N * M>();

    // gemm using inner product
    for (int n = 0; n < N; n++) {

        for (int m = 0; m < M; m++) {

            T c_n_m = 0;

            for (int k = 0; k < K; k++) {

                T a_n_k = a.buffer->at(n * K + k);
                T b_k_m = b.buffer->at(k * M + m);

                c_n_m += a_n_k * b_k_m;
            }

            buffer->at(n * M + m) = c_n_m;
        }
    }

    return Tensor<T, N, M>(buffer);
};

template<typename T, int N, int M>
Tensor<T, N, M> add(
    Tensor<T, N, M>& a,
    Tensor<T, N, M>& b
) {

    std::array<T, N * M>* buffer = new std::array<T, N * M>();

    for (int i = 0; i < a.buffer->size(); i++) {

        buffer->at(i) = a.buffer->at(i) + b.buffer->at(i);
    }

    return Tensor<T, N, M>(buffer);
};

