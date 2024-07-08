# main: ./src/logger.cpp ./src/ml.cpp ./src/main.cpp

# 	clear

# 	rm -rf ./build

# 	mkdir ./build

# 	clang++ ./src/logger.cpp ./src/ml.cpp ./src/main.cpp -o ./build/main -std=c++20

# 	./build/main

main: ./src/main.cpp

	clear

	rm -rf ./build

	mkdir ./build

	clang++ ./src/main.cpp -o ./build/main -std=c++20

	./build/main
