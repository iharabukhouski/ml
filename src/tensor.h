#pragma once

#include <initializer_list>
#include <array>
#include "./backend/cpu/cpu2.h"

template<
    typename T,
    int N, // number of rows
    int M // number of columns
>
class Tensor {

    public:

        Buffer<T, N, M>* buffer;
        Buffer<T, N, M>* grad;

        // op
        // args

        const int _N = N;
        const int _M = M;

        Tensor(
            std::initializer_list<T> items
        ) {

            // USAGE:
            //
            // Tensor<float, 2, 2> a = Tensor<float, 2, 2>({
            //     2, 3,
            //     4, 5,
            // });

            this->buffer = new Buffer<T, N, M>();

            int i = 0;

            for (int item : items) {

                this->buffer->at(i) = item;

                i++;
            }

            this->grad = new Buffer<T, N, M>();
        };

        Tensor(
            Buffer<T, N, M>* buffer
        ) {

            this->buffer = buffer;
            this->grad = new Buffer<T, N, M>();
        };

        Tensor<T, N, M> operator+(
            Tensor<T, N, M> b
        ) {

            AddOp<T, N, M>* addOp = new AddOp<T, N, M>();

            addOp->forward(
                this->buffer,
                b.buffer
            );

            return b;
        };

        static Tensor<T, N, M> randn() {

            Buffer<T, N, M>* buffer = randn1<T, N, M>();

            return Tensor(buffer);
        }
};
