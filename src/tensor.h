#include "op.h"
#include "./backend/backend.h"

template<
    int N,
    int M,
    typename T = float
>
class tensor {

    public:

        backend::buffer data;
        backend::buffer grad;
        op op;

        tensor() {

        }

        void backward() {

        }
};

template<
    int N,
    int M,
    typename T = float
>
tensor<N, M, T> zeros() {

    return tensor();
}

template<
    int N,
    int M,
    typename T = float
>
tensor<N, M, T> ones() {

    return tensor();
};

template<
    int N,
    int M,
    typename T = float
>
tensor<N, M, T> full(
    float a
) {

    return tensor();
};

template<
    int N,
    int M,
    typename T = float
>
tensor<N, M, T> randn() {

    return tensor();
}

template<
    int N,
    int M,
    typename T = float
>
tensor<N, M, T> add(
    tensor<N, M, T> a,
    tensor<N, M, T> b
) {

    auto c = tensor();

    c.op = backend::add();
    c.data = c.op.forward(
        a.data,
        b.data
    );

    return c;
}

template<
    int N,
    int M,
    typename T = float
>
tensor<N, M, T> sub(
    tensor<N, M, T> a,
    tensor<N, M, T> b
) {

    return tensor();
}

template<
    int N,
    int M,
    typename T = float
>
tensor<N, M, T> mul(
    tensor<N, M, T> a,
    tensor<N, M, T> b
) {

    return tensor();
}

template<
    int N,
    int M,
    typename T = float
>
tensor<N, M, T> matmul(
    tensor<N, M, T> a,
    tensor<N, M, T> b
) {

    return tensor();
}

template<
    int N,
    int M,
    typename T = float
>
tensor<N, M, T> relu(
    tensor<N, M, T> a
) {

    return tensor();
}

template<
    int N,
    int M,
    typename T = float
>
tensor<M, N, T> transpose(
    tensor<N, M, T> a
) {

    return tensor();
}
