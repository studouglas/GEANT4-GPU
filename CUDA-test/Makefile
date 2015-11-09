all:
	gcc -c *.cpp
	nvcc -c *.cu
	nvcc -o main *.o

run: build
	./main

clean:
	rm -rf *.o
	rm main