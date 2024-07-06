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

    private:

        std::array<T, N * M>* _buffer = new std::array<T, N * M>();

    public:

        T& at(
            int i
        ) {

            return this->_buffer->at(i);
        }

        int size() {

            return this->_buffer->size();
        }

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

            for (int i = 0; i < N * M; i++) {

                buffer->at(i) = a->at(i) + b->at(i);
            }

            return buffer;
        }

        std::array<Buffer<T, N, M>*, 2> backward(
            Buffer<T, N, M>* grad
        ) {

            return std::array<Buffer<T, N, M>*, 2>({
                grad,
                grad
            });
        }
};

namespace device {

    template<
        typename T,
        int N,
        int M
    >
    Buffer<T, N, M>* zeros() {

        Buffer<T, N, M>* buffer = new Buffer<T, N, M>();

        for (int i = 0; i < N * M; i++) {

            buffer->at(i) = 0;
        }

        return buffer;
    };

    template<
        typename T,
        int N,
        int M
    >
    Buffer<T, N, M>* ones() {

        Buffer<T, N, M>* buffer = new Buffer<T, N, M>();

        for (int i = 0; i < N * M; i++) {

            buffer->at(i) = 1;
        }

        return buffer;
    };

    template<
        typename T,
        int N,
        int M
    >
    Buffer<T, N, M>* randn() {

        Buffer<T, N, M>* buffer = new Buffer<T, N, M>();

        std::default_random_engine generator(std::time(0));
        std::normal_distribution<T> distribution(0, 1);

        for (int i = 0; i < N * M; i++) {

            buffer->at(i) = distribution(generator);
        }

        return buffer;
    };

}
