################################################################################
docker_cmd := podman

all:
	echo "Nothing to do"


_wheels_build_cmd:
	$(docker_cmd) run\
		--rm\
		-v ./wheelhouse:/project/wheelhouse\
		-v ./pmpc:/project/pmpc\
		-v ./scripts:/project/scripts\
		-v ./setup.py:/project/setup.py\
		pmpc

################################################################################

clean:
	rm -rf build
	rm -rf dist
	rm -rf PMPC.jl/build
	rm -rf PMPC.jl/pmpcjl/lib
	rm -rf PMPC.jl/pmpcjl/share
	rm -rf PMPC.jl/pmpcjl/pmpc.egg-info
	rm -rf PMPC.jl/pmpcjl/*.so

container: clean
	$(docker_cmd) build --build-arg ARCH=$(shell uname -m) -t pmpc .

is_pmpc_container_built := $(shell podman images | grep pmpc)
ifeq ($(is_pmpc_container_built),)
wheels: container _wheels_build_cmd
else
wheels: _wheels_build_cmd
endif

install:
	pip install .

install_static:
	python3 scripts/install_static.py

install_dynamic:
	python3 scripts/install_dynamic.py

################################################################################
