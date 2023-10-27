ARG ARCH

FROM quay.io/pypa/manylinux_2_28_${ARCH}

COPY . /project
WORKDIR /project

RUN yes | yum update
RUN yes | yum install curl wget

# configure julia
RUN curl -fsSL https://install.julialang.org | sh -s -- --default-channel 1.6 -y
RUN source ~/.bashrc

# make sure dynamic libraries are findable with `locate`
RUN yes | yum install mlocate
RUN updatedb

RUN mkdir /root/mosek
COPY ./mosek.lic /root/mosek/mosek.lic

# create the julia binaries
RUN rm -rf /project/PMPC.jl/build
RUN ~/.juliaup/bin/julia /project/PMPC.jl/scripts/build_pmpc_lib.jl

WORKDIR /project

#RUN /project/scripts/build_for_all.sh
CMD ["sh", "/project/scripts/build_for_all.sh"]
