all:
	g++ -O2 -Wall -shared -fPIC -std=c++11 $(shell python3 -m pybind11 --includes)\
		$(shell python3-config --includes)\
	  module.cpp -o pmpcjl$(shell python3-config --extension-suffix)\
		$(shell python3-config --ldflags)\
	  -I./include -L./lib -ljulia -lPMPC

