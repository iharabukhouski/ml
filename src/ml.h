#pragma once

#include <random>
#include <ctime>
#include <string>
#include <vector>

class Tensor {

    // private:

    public:

        // std::vector<float> data;
        float data;

        // operation that produced this tensor
        char op;

        float grad;

        // args to the operation that produced this tensor
        std::vector<Tensor*>* args;

    // private:

    //     template<typename T>
    //     void variadic_constructor(T t);

    //     template<typename T, typename ...Args>
    //     void variadic_constructor(T t, Args ...args);

    public:



        // template<typename ...Args>
        // Tensor(Args ...args);

        Tensor(
            float data
        );

        // TODO: see if it is possible to get rid of this constructor
        Tensor(
            // std::vector<float> data,
            float data,
            char op,
            std::vector<Tensor*>* args
        );

        Tensor operator+(float _b);

        Tensor operator+(Tensor& b);

        Tensor operator-(Tensor& b);

        Tensor operator*(Tensor& b);

        Tensor operator/(Tensor& b);

        void zero_grad();

        void backward();

        std::string __repr__();

};

void zero_grad(
    Tensor* a
);

void backward(
    Tensor* a
);

std::vector<Tensor*>* toposort(
    Tensor* a
);

template<typename T, int n_rows, int n_cols>
std::array<T, n_rows * n_cols>* randn() {

    std::array<T, n_rows * n_cols>* arr = new std::array<T, n_rows * n_cols>();

    std::default_random_engine generator(std::time(0));
    std::normal_distribution<float> distribution(0, 1);

    for (int i = 0; i < n_rows * n_cols; i++) {

        arr->at(i) = distribution(generator);
    }

    return arr;
};
