#pragma once

#include <iostream>
#include <initializer_list>
#include <optional>
#include <set>
#include <array>
#include "logger.h"
#include "./backend/cpu/cpu2.h"

template<
    typename T,
    int N, // number of rows
    int M // number of columns
>
class Tensor {

    private:

    public:

        Tensor(
            Buffer<T, N, M>* data
        ) {

            this->data = data;
            this->grad = new Buffer<T, N, M>();
        };

        Tensor() {

            // no-op
            this->grad = new Buffer<T, N, M>();
        };

        Buffer<T, N, M>* data;
        Buffer<T, N, M>* grad;

        Op<T, N, M>* op = NULL;
        std::vector<Tensor<T, N, M>*>* args = NULL;

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

            this->data = new Buffer<T, N, M>();

            int i = 0;

            for (int item : items) {

                this->data.at(i) = item;

                i++;
            }

            this->grad = new Buffer<T, N, M>();
        };

        Tensor<T, N, M> operator+(
            Tensor<T, N, M> &b
        ) {

            Tensor<T, N, M> c = Tensor<T, N, M>();

            c.args = new std::vector<Tensor<T, N, M>*>({
                this,
                &b
            });

            c.op = new AddOp<T, N, M>();

            c.data = c.op->forward(
                this->data,
                b.data
            );

            return c;
        };

        // operator Tensor<T, M, N>() {

        //     std::cout << "hi";

        //     return this;
        // }

        static Tensor<T, N, M> zeros() {

            Buffer<T, N, M>* data = device::zeros<T, N, M>();

            return Tensor<T, N, M>(data);
        }

        static Tensor<T, N, M> ones() {

            Buffer<T, N, M>* data = device::ones<T, N, M>();

            return Tensor<T, N, M>(data);
        }

        static Tensor<T, N, M> randn() {

            Buffer<T, N, M>* data = device::randn<T, N, M>();

            return Tensor<T, N, M>(data);
        }

        static Tensor<T, M, N> transpose(
            Tensor<T, N, M>& t
        ) {

            Tensor<T, M, N> _t = Tensor<T, M, N>();

            _t.data = (Buffer<T, M, N>*) t.data;
            _t.data->stride = M;
            _t.grad = (Buffer<T, M, N>*) t.grad;
            _t.grad->stride = M;
            // _t.op = t0.op;
            // _t.args = t0.args;

            return _t;
        }
};

template<
    int N, // number of rows
    int M, // number of columns
    typename T = float
>
Tensor<T, N, M> zeros() {

    Buffer<T, N, M>* data = device::zeros<T, N, M>();

    return Tensor<T, N, M>(data);
}

template<
    int N, // number of rows
    int M, // number of columns
    typename T = float
>
Tensor<T, N, M> ones() {

    Buffer<T, N, M>* data = device::ones<T, N, M>();

    return Tensor<T, N, M>(data);
}

template<
    int N, // number of rows
    int M, // number of columns
    typename T = float
>
Tensor<T, N, M> randn() {

    Buffer<T, N, M>* data = device::randn<T, N, M>();

    return Tensor<T, N, M>(data);
}

template<
    int N, // number of rows
    int M, // number of columns
    typename T = float
>
Tensor<T, M, N> transpose(
    Tensor<T, N, M>& t
) {

    Tensor<T, M, N> _t = Tensor<T, M, N>();

    _t.data = (Buffer<T, M, N>*) t.data;
    _t.data->stride = M;
    _t.data->contiguous = false;
    _t.grad = (Buffer<T, M, N>*) t.grad;
    _t.grad->stride = M;
    _t.grad->contiguous = false;
    // _t.op = t0.op;
    // _t.args = t0.args;

    return _t;
}

template<
    int N, // number of rows
    int M, // number of columns
    typename T = float
>
Tensor<T, N, M> add(
    Tensor<T, N, M>& a,
    Tensor<T, N, M>& b
) {

    auto c = Tensor<T, N, M>();

    c.args = new std::vector<Tensor<T, N, M>*>({
        &a,
        &b
    });

    c.op = new AddOp<T, N, M>();

    c.data = c.op->forward(
        a.data,
        b.data
    );

    return c;
}

template<
    int N, // number of rows
    int M, // number of columns
    typename T = float
>
Tensor<T, N, M> add(
    Tensor<T, N, M>& a,
    T b
) {

    auto _b = Tensor<T, N, M>();

    _b.data = device::full<T, N, M>(b);

    auto c = Tensor<T, N, M>();

    c.args = new std::vector<Tensor<T, N, M>*>({
        &a,
        &_b
    });

    c.op = new AddOp<T, N, M>();

    c.data = c.op->forward(
        a.data,
        _b.data
    );

    return c;
}

template<
    int N, // number of rows
    int M, // number of columns
    typename T = float
>
Tensor<T, N, M> mul(
    Tensor<T, N, M>& a,
    T b
) {

    auto _b = Tensor<T, N, M>();

    _b.data = device::full<T, N, M>(b);

    auto c = Tensor<T, N, M>();

    c.args = new std::vector<Tensor<T, N, M>*>({
        &a,
        &_b
    });

    c.op = new MulOp<T, N, M>();

    c.data = c.op->forward(
        a.data,
        _b.data
    );

    return c;
}

template<
    int N, // number of rows
    int M, // number of columns
    typename T = float
>
Tensor<T, N, M> mul(
    Tensor<T, N, M>& a,
    Tensor<T, N, M>& b
) {

    auto c = Tensor<T, N, M>();

    c.args = new std::vector<Tensor<T, N, M>*>({
        &a,
        &b
    });

    c.op = new MulOp<T, N, M>();

    c.data = c.op->forward(
        a.data,
        b.data
    );

    return c;
}

// template<
//     typename T,
//     int N,
//     int M
// >
// Tensor<T, M, N> transpose(
//     Tensor<T, N, M>& t
// ) {

//     Tensor<T, M, N> _t = Tensor<T, M, N>();

//     _t.stride = t.stride;
//     _t.data = t.data;
//     _t.grad = t.grad;
//     _t.op = t.op;
//     _t.args = t.args;

//     return _t;
// }

template<
    typename T,
    int N,
    int M
>
void _toposort(
    std::set<Tensor<T, N, M>*>* visited,
    std::vector<Tensor<T, N, M>*>* sorted,
    Tensor<T, N, M>* a
) {

    if (visited->count(a) == 0) {

        visited->insert(a);

        if (a->args != NULL && a->args->size() != 0) {

            for (int i = 0; i < a->args->size(); i++) {

                _toposort<T, N, M>(
                    visited,
                    sorted,
                    a->args->at(i)
                );
            }

            sorted->push_back(a);
        }
    }
};

template<
    typename T,
    int N,
    int M
>
std::vector<Tensor<T, N, M>*>* toposort(
    Tensor<T, N, M>* a
) {

    std::set<Tensor<T, N, M>*>* visited = new std::set<Tensor<T, N, M>*>();
    std::vector<Tensor<T, N, M>*>* sorted = new std::vector<Tensor<T, N, M>*>();

    _toposort(
        visited,
        sorted,
        a
    );

    delete visited;

    std::reverse(sorted->begin(), sorted->end());

    return sorted;
};

template<
    typename T,
    int N,
    int M
>
void backward(
    Tensor<T, N, M>& t
) {

    t.grad = device::ones<T, N, M>();

    for (Tensor<T, N, M>* _t : *toposort(&t)) {

        // print(*_t);

        std::array<Buffer<T, N, M>*, 2> grads = _t->op->backward(_t->grad);

        for (int j = 0; j < grads.size(); j++) {

            _t->args->at(j)->grad = grads.at(j);

            // print(*_t->args->at(j));
        }
    }
}
