#pragma once

#include <initializer_list>
#include <array>
#include <type_traits>

enum DType {
    f16,
    f32
};

template<
    typename T,
    int N, // number of rows
    int M, // number of columns
    DType foo = f32
>
class Tensor {

    typedef typename std::conditional<foo == f32, float, int>::type dtype;

    public:

        std::array<T, N * M>* buffer;

        // const auto dtype = T;
        const int _N = N;
        const int _M = M;

        // DType dtype;

        Tensor(
            std::initializer_list<dtype> items
        ) {

            // USAGE:
            //
            // Tensor<float, 2, 2> a = Tensor<float, 2, 2>({
            //     2, 3,
            //     4, 5,
            // });

            std::array<dtype, N * M>* buffer = new std::array<dtype, N * M>();

            int i = 0;

            for (int item : items) {

                buffer->at(i) = item;

                i++;
            }

            this->buffer = buffer;
        };

        Tensor(
            std::array<dtype, N * M>* buffer
        ) {

            this->buffer = buffer;
        };
};
