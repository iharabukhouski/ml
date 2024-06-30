#pragma once

template <typename T>
void print(T t) {

    std::cout << t << std::endl;
};

template<typename T, typename... Args>
void print(T t, Args... args) {

    std::cout << t <<std::endl;

    print(args...);
};
