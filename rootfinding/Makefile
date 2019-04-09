executables := $(shell find . -maxdepth 1 -name "*.o")

all:
		g++ -Wall markov_chain_montecarlo.cpp -o mcmc.o \
		-x c++ rootfinder.cpp \
		-x c++ geometry.cpp \
		-x c /usr/local/include/cubature/hcubature.c

clean:
		-rm $(executables)
