################################################################################
docker_cmd := podman

# Get the system's architecture
ARCH := $(shell uname -m)
# Get the system type
OS := $(shell uname -s)

all:
ifeq ($(ARCH),aarch64)
ifeq ($(OS),Linux)
container_image := fedora:37
else ifeq ($(OS),Darwin)
container_image := ""
endif
else ifeq ($(ARCH),x86_64)
ifeq ($(OS),Linux)
container_image := quay.io/pypa/manylinux_2_28_x86_64
else ifeq ($(OS),Darwin)
container_image := ""
endif
endif

all:
	echo "Nothing to do"


_wheels_build_cmd:
	mkdir -p ./wheelhouse
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

container:
	$(docker_cmd) build\
		--build-arg IMAGE=$(container_image)\
		--build-arg ARCH=$(shell uname -m)\
		-t pmpc .

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
