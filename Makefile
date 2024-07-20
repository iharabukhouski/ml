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

	# [ -f main.o ] && rm main.o
	# [ -f cuda.o ] && rm cuda.o

program: main.o cuda.o

	clang++ -L/usr/local/cuda-12/lib64 cuda.o main.o -o program -lcudart -lcublas

main.o: ./src/main.cpp

	clang++ -c ./src/main.cpp -o main.o

cuda.o: ./src/backend/cuda/matmul/cublas.cu

	nvcc -c ./src/backend/cuda/matmul/cublas.cu -o cuda.o
