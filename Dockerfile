ARG IMAGE

#FROM quay.io/pypa/manylinux_2_28_${ARCH}
#FROM fedora:35
FROM ${IMAGE}

COPY . /project
WORKDIR /project

RUN yum update -y
RUN yum install -y curl wget 

# install build tools if we're on fedora
RUN yum install -y g++ gcc || true
RUN yum install -y patchelf || true
RUN yum install -y python3.7 || true
RUN yum install -y python3.7-devel || true
RUN yum install -y python3.8 || true
RUN yum install -y python3.8-devel || true
RUN yum install -y python3.9 || true
RUN yum install -y python3.9-devel || true
RUN yum install -y python3.10 || true
RUN yum install -y python3.10-devel || true
RUN yum install -y python3.11 || true
RUN yum install -y python3.11-devel || true
RUN yum install -y python3.12 || true
RUN yum install -y python3.12-devel || true

# configure julia
RUN curl -fsSL https://install.julialang.org | sh -s -- --default-channel 1.6.7 -y
RUN source ~/.bashrc

# make sure dynamic libraries are findable with `locate`
RUN yum install -y mlocate || true
RUN updatedb || true

RUN mkdir /root/mosek
COPY ./mosek.lic /root/mosek/mosek.lic

# create the julia binaries
RUN rm -rf /project/PMPC.jl/build
RUN ~/.juliaup/bin/julia /project/PMPC.jl/scripts/build_pmpc_lib.jl

WORKDIR /project

#RUN /project/scripts/build_for_all.sh
CMD ["sh", "/project/scripts/build_for_all.sh"]
