INCLUDE	= -I/usr/local/include/eigen3

main: main.cc legendre_decomposition.h
	g++ -O3 $(INCLUDE) -std=c++11 -march=native main.cc -o ld
	# icpc -O3 -fast -mkl $(INCLUDE) -std=c++11 -march=native main.cc -o ltd
	# icpc -O3 -fast -mkl:sequential $(INCLUDE) -std=c++11 -march=native main.cc -o ltd
