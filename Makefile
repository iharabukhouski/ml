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

all: clean program

	./program

clean:

	clear

	rm -f cublas.o cuda.o main.o

program: cublas.o cuda.o main.o 

	clang++ -L/usr/local/cuda-12/lib64 cublas.o cuda.o main.o -o program -lcudart -lcublas

main.o: ./src/main.cpp

	clang++ -c ./src/main.cpp -o main.o

cuda.o: ./src/backend/cuda/matmul/naive.cu

	nvcc -c ./src/backend/cuda/matmul/naive.cu -o cuda.o

cublas.o: ./src/backend/cuda/matmul/cublas.cu

	nvcc -c ./src/backend/cuda/matmul/cublas.cu -o cublas.o
