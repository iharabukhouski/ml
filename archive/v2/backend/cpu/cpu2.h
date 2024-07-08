#include <iostream>
#include <random>
#include <ctime>
// #include "../../op.h"

std::default_random_engine generator(std::time(0));

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

        int stride = 1;
        bool contiguous = true;

        const int _N = N;
        const int _M = M;

        operator Buffer<T, M, N>*() {

            auto b = new Buffer<T, M, N>();

            b->_buffer = this->_buffer;

            return this;
        }

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
class Op {

    public:

    virtual Buffer<T, N, M>* forward(
            Buffer<T, N, M>* a,
            Buffer<T, N, M>* b
        ) = 0;

    virtual std::array<Buffer<T, N, M>*, 2> backward(
            Buffer<T, N, M>* grad
        ) = 0;

};

template<
    typename T,
    int N,
    int M
>
class MatMulOp : public Op<T, N, M> {

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


template<
    typename T,
    int N,
    int M
>
class MulOp : public Op<T, N, M> {

    // hadamar product

    public:

        Buffer<T, N, M>* forward(
            Buffer<T, N, M>* a,
            Buffer<T, N, M>* b
        ) {

            Buffer<T, N, M>* buffer = new Buffer<T, N, M>();

            for (int i = 0; i < N * M; i++) {

                buffer->at(i) = a->at(i) * b->at(i);
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

template<
    typename T,
    int N,
    int M
>
class AddOp : public Op<T, N, M> {

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
    Buffer<T, N, M>* full(
        T t
    ) {

        Buffer<T, N, M>* buffer = new Buffer<T, N, M>();

        for (int i = 0; i < N * M; i++) {

            buffer->at(i) = t;
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

        std::normal_distribution<T> distribution(0, 1);

        for (int i = 0; i < N * M; i++) {

            buffer->at(i) = distribution(generator);
        }

        return buffer;
    };

}

template<
    int N,
    int M,
    typename T
>
Buffer<T, N, M>* add(
    Buffer<T, N, M>* a,
    T b
) {

    Buffer<T, N, M>* _b = device::full<T, N, M>(b);

    Buffer<T, N, M>* buffer = new Buffer<T, N, M>();

    for (int i = 0; i < N * M; i++) {

        buffer->at(i) = a->at(i) + _b->at(i);
    }

    return buffer;
}

template<
    int N,
    int M,
    typename T
>
Buffer<T, N, M>* add(
    Buffer<T, N, M>* a,
    Buffer<T, N, M>* b
) {

    Buffer<T, N, M>* buffer = new Buffer<T, N, M>();

    for (int i = 0; i < N * M; i++) {

        buffer->at(i) = a->at(i) + b->at(i);
    }

    return buffer;
}

template<
    int N,
    int M,
    typename T
>
Buffer<T, N, M>* mul(
    Buffer<T, N, M>* a,
    T b
) {

    Buffer<T, N, M>* _b = device::full<T, N, M>(b);

    Buffer<T, N, M>* buffer = new Buffer<T, N, M>();

    for (int i = 0; i < N * M; i++) {

        buffer->at(i) = a->at(i) * _b->at(i);
    }

    return buffer;
}

template<
    int N,
    int M,
    typename T
>
Buffer<T, N, M>* mul(
    Buffer<T, N, M>* a,
    Buffer<T, N, M>* b
) {

    Buffer<T, N, M>* buffer = new Buffer<T, N, M>();

    for (int i = 0; i < N * M; i++) {

        buffer->at(i) = a->at(i) * b->at(i);
    }

    return buffer;
}
