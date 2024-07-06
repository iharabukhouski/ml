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

        Tensor() {

            // no-op
        };

        Tensor(
            Buffer<T, N, M>* buffer
        ) {

            this->buffer = buffer;
            this->grad = new Buffer<T, N, M>();
        };

    public:

        Buffer<T, N, M>* buffer;
        Buffer<T, N, M>* grad;

        AddOp<T, N, M>* op = NULL;
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

            this->buffer = new Buffer<T, N, M>();

            int i = 0;

            for (int item : items) {

                this->buffer.at(i) = item;

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

            c.buffer = c.op->forward(
                this->buffer,
                b.buffer
            );

            return c;
        };

        static Tensor<T, N, M> zeros() {

            Buffer<T, N, M>* buffer = device::zeros<T, N, M>();

            return Tensor<T, N, M>(buffer);
        }

        static Tensor<T, N, M> ones() {

            Buffer<T, N, M>* buffer = device::ones<T, N, M>();

            return Tensor<T, N, M>(buffer);
        }

        static Tensor<T, N, M> randn() {

            Buffer<T, N, M>* buffer = device::randn<T, N, M>();

            return Tensor<T, N, M>(buffer);
        }
};

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

    std::vector<Tensor<T, N, M>*>* sorted = toposort(&t);

    for (int i = 0; i < sorted->size(); i++) {

        Tensor<T, N, M>* _t = sorted->at(i);

        std::array<Buffer<T, N, M>*, 2> grads = _t->op->backward(_t->grad);

        for (int j = 0; j < grads.size(); j++) {

            _t->args->at(j)->grad = grads.at(j);
        }
    }
}
