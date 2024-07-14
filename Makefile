# main: ./src/logger.cpp ./src/ml.cpp ./src/main.cpp

# 	clear

# 	rm -rf ./build

# 	mkdir ./build

# 	clang++ ./src/logger.cpp ./src/ml.cpp ./src/main.cpp -o ./build/main -std=c++20

# 	./build/main

# main: ./src/main.cpp

# 	clear

# 	rm -rf ./build

# 	mkdir ./build

# 	clang++ ./src/main.cpp -o ./build/main -std=c++20

# 	./build/main

all: program

	./program

program: main.o cuda.o

	clang++ -L/usr/local/cuda-12/lib64 cuda.o main.o -o program -lcudart

main.o: main.cpp

	clang++ -c main.cpp

cuda.o: cuda.cu

	nvcc -c cuda.cu
