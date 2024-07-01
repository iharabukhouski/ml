#include <iostream>
#include <random>
#include <ctime>
#include "../../op.h"

template<
    typename T,
    int N,
    int M
>
class Buffer {

    // represents device buffer

    public:

        std::array<T, N * M>* ptr = new std::array<T, N * M>();

};

template<
    typename T,
    int N,
    int M
>
class AddOp {

    // represents add op for this particular device

    public:

        Buffer<T, N, M>* forward(
            Buffer<T, N, M>* a,
            Buffer<T, N, M>* b
        ) {

            Buffer<T, N, M>* buffer = new Buffer<T, N, M>();

            for (int i = 0; i < a->ptr->size(); i++) {

                buffer->ptr->at(i) = a->ptr->at(i) + b->ptr->at(i);
            }

            return buffer;
        }

        Buffer<T, N, M> backward() {

            // TODO
        }
};

template<
    typename T,
    int N,
    int M
>
Buffer<T, N, M>* randn1() {

    Buffer<T, N, M>* buffer = new Buffer<T, N, M>();

    std::default_random_engine generator(std::time(0));
    std::normal_distribution<T> distribution(0, 1);

    for (int i = 0; i < N * M; i++) {

        buffer->ptr->at(i) = distribution(generator);
    }

    return buffer;
};
