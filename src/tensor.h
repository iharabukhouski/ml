#pragma once

#include <initializer_list>
#include <array>

enum DType {
    f16,
    f32
};

template<
    typename T,
    int N, // number of rows
    int M // number of columns
>
class Tensor {

    public:

        std::array<T, N * M>* buffer;

        // const auto dtype = T;
        const int _N = N;
        const int _M = M;

        DType dtype;

        Tensor(
            std::initializer_list<T> items,
            DType dtype = f32
        ) {

            // USAGE:
            //
            // Tensor<float, 2, 2> a = Tensor<float, 2, 2>({
            //     2, 3,
            //     4, 5,
            // });

            std::array<T, N * M>* buffer = new std::array<T, N * M>();

            int i = 0;

            for (int item : items) {

                buffer->at(i) = item;

                i++;
            }

            this->buffer = buffer;
        };

        Tensor(
            std::array<T, N * M>* buffer,
            DType dtype = f32
        ) {

            this->buffer = buffer;
            this->dtype = dtype;
        };
};
