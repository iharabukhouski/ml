#include <set>
#include <vector>
#include "tensor.h"

void _toposort(
    std::set<Tensor*>* visited,
    std::vector<Tensor*>* sorted,
    Tensor* a
) {

    if (visited->count(a) == 0) {

        visited->insert(a);

        if (a->args != NULL && a->args->size() != 0) {

            for (int i = 0; i < a->args->size(); i++) {

                _toposort(
                    visited,
                    sorted,
                    a->args->at(i)
                );
            }
        }

        sorted->push_back(a);
    }
};

std::vector<Tensor*>* toposort(
    Tensor* a
) {

    std::set<Tensor*>* visited = new std::set<Tensor*>();
    std::vector<Tensor*>* sorted = new std::vector<Tensor*>();

    _toposort(
        visited,
        sorted,
        a
    );

    delete visited;

    std::reverse(sorted->begin(), sorted->end());

    return sorted;
};


void zero_grad(
    Tensor* a
) {

    std::vector<Tensor*>* sorted = toposort(a);

    for (int i = 0; i < (sorted)->size(); i++) {

        (*sorted)[i]->grad = 0;
    }

    delete sorted;
};

void backward() {

    if (this->args != NULL && this->args->size() != 0) {

        if (this->op == '+') {

            plus_backward(
                this->grad,
                (*this->args)[0],
                (*this->args)[1]
            );
        }

        if (this->op == '*') {

            multiply_backward(
                this->grad,
                (*this->args)[0],
                (*this->args)[1]
            );
        }
    }
}

void backward(
    Tensor* a
) {

    std::vector<Tensor*>* sorted = toposort(a);

    (*sorted)[0]->grad = 1;

    for (int i = 0; i < (sorted)->size(); i++) {

        (*sorted)[i]->backward();
    }

    delete sorted;
}
