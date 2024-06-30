#include <iostream>
#include <string>
#include <array>
#include <vector>
#include <set>
#include "logger.h"
#include "ml.h"

// template<typename T>
// void Tensor::variadic_constructor(T t) {

//     data.push_back(t);
// };

// template<typename T, typename ...Args>
// void Tensor::variadic_constructor(T t, Args ...args) {

//     data.push_back(t);

//     variadic_constructor(args...);
// };

// template<typename ...Args>
// Tensor::Tensor(Args ...args) {

//     variadic_constructor(args...);
// }

Tensor::Tensor(
    float data
) {

    this->data = data;
    this->op = (char)NULL;
    this->args = NULL;
    this->grad = 0;
}

// Tensor::Tensor(

// ) {

// }

Tensor::Tensor(
    // std::vector<float> data,
    float data,
    char op,
    std::vector<Tensor*>* args
) {

    this->data = data;
    this->op = op;
    this->args = args;
    this->grad = 0;
};

// Tensor Tensor::operator+(float _b) {

//     std::vector<float> _data(data.size());
//     std::copy(data.begin(), data.end(), _data.begin());

//     Tensor b = Tensor(_b);

//     for (int i = 0; i < data.size(); i++) {

//         _data[i] = data[i] + b.data[i];
//     }

//     std::vector<Tensor*> _args;

//     _args.push_back(this);
//     _args.push_back(&b);

//     return Tensor(
//         _data,
//         '+',
//         _args
//     );
// }

// Tensor Tensor::operator+(Tensor& b) {

//     std::vector<float> data(this->data.size());
//     std::copy(this->data.begin(), this->data.end(), data.begin());

//     for (int i = 0; i < this->data.size(); i++) {

//         data[i] = this->data[i] + b.data[i];
//     }

//     std::vector<Tensor*> _args = std::vector<Tensor*>{};

//     args.push_back(this);
//     args.push_back(&b);

//     return Tensor(
//         data,
//         '+',
//         args
//     );
// }

Tensor Tensor::operator+(
    Tensor& b
) {

    // std::vector<float> data(this->data.size());
    // std::copy(this->data.begin(), this->data.end(), data.begin());

    // for (int i = 0; i < this->data.size(); i++) {

    //     data[i] = this->data[i] + b.data[i];
    // }

    float data = this->data + b.data;

    std::vector<Tensor*>* args = new std::vector<Tensor*>();

    args->push_back(this);
    args->push_back(&b);

    return Tensor(
        data,
        '+',
        args
    );
}

// Tensor Tensor::operator-(Tensor& b) {

//     std::vector<float> _data(data.size());
//     std::copy(data.begin(), data.end(), _data.begin());

//     for (int i = 0; i < data.size(); i++) {

//         _data[i] = data[i] - b.data[i];
//     }

//     std::vector<Tensor*> _args;

//     _args.push_back(this);
//     _args.push_back(&b);

//     return Tensor(
//         _data,
//         '-',
//         _args
//     );
// }

// Tensor Tensor::operator*(Tensor& b) {

//     std::vector<float> _data(data.size());
//     std::copy(data.begin(), data.end(), _data.begin());

//     for (int i = 0; i < data.size(); i++) {

//         _data[i] = data[i] * b.data[i];
//     }

//     std::vector<Tensor*> _args = std::vector<Tensor*>{};

//     _args.push_back(this);
//     _args.push_back(&b);

//     return Tensor(
//         _data,
//         '*',
//         _args
//     );
// }

Tensor Tensor::operator*(Tensor& b) {

    float data = this->data * b.data;

    std::vector<Tensor*>* args = new std::vector<Tensor*>();

    args->push_back(this);
    args->push_back(&b);

    return Tensor(
        data,
        '*',
        args
    );
}

// Tensor Tensor::operator/(Tensor& b) {

//     std::vector<float> _data(data.size());
//     std::copy(data.begin(), data.end(), _data.begin());

//     for (int i = 0; i < data.size(); i++) {

//         _data[i] = data[i] / b.data[i];
//     }

//     std::vector<Tensor*> _args;

//     _args.push_back(this);
//     _args.push_back(&b);

//     return Tensor(
//         _data,
//         '/',
//         _args
//     );
// }

void zero_grad(
    Tensor* a
) {

    std::vector<Tensor*>* sorted = toposort(a);

    for (int i = 0; i < (sorted)->size(); i++) {

        (*sorted)[i]->grad = 0;
    }

    delete sorted;
}

void plus_backward(
    float grad,
    Tensor* a,
    Tensor* b
) {

    a->grad = 1 * grad;
    b->grad = 1 * grad;
}

void multiply_backward(
    float grad,
    Tensor* a,
    Tensor* b
) {

    a->grad = b->data * grad;
    b->grad = a->data * grad;
}

void Tensor::backward() {

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

std::string Tensor::__repr__() {

    // std::string string = "[{data}] [{grad}]";

    // // string.replace(1, 6, std::to_string(this->data[0]));
    // string.replace(1, 6, std::to_string(this->data));
    // string.replace(11, 11, std::to_string(this->grad));

    // return string;

    if (this->args != NULL && this->args->size() != 0) {

        return std::to_string(this->args->size()) + " |  " + std::to_string(this->args->at(0)->data) + " " + (std::string)&this->op + " " + std::to_string(this->args->at(1)->data) + " [" + std::to_string(this->data) + "] [" + std::to_string(this->grad) + "]";
    } else {

        return "0  |  [" + std::to_string(this->data) + "] [" + std::to_string(this->grad) + "]";
    }

    // return "";
    
    // return std::to_string(this->args->at(0)->data);
}

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
