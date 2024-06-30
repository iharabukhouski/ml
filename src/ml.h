#pragma once

#include <array>
#include <random>
#include <ctime>
#include <string>
#include <vector>

// template<typename T, int n_rows, int n_cols>
class Tensor {

    // private:

    public:

        std::array<float, 3 * 2>* buffer;
        // std::array<T, n_rows * n_cols>* _buffer;

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

        Tensor(
            std::array<float, 3 * 2>* buffer
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

// template<typename T, int n_rows, int n_cols>
// Tensor<T, n_rows, n_cols>* randn() {

//     std::array<T, n_rows * n_cols>* buffer = new std::array<T, n_rows * n_cols>();

//     std::default_random_engine generator(std::time(0));
//     std::normal_distribution<T> distribution(0, 1);

//     for (int i = 0; i < n_rows * n_cols; i++) {

//         buffer->at(i) = distribution(generator);
//     }

//     return Tensor(buffer);
// };
