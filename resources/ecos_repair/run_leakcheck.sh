#!/bin/bash

valgrind --smc-check=all-non-file --leak-check=full \
  --suppressions=data/valgrind-julia.supp julia data/direct.jl 2>&1 \
  | tee data/direct.txt

#valgrind --smc-check=all-non-file --leak-check=full \
#  --suppressions=data/valgrind-julia.supp julia data/lsocp1.jl 2>&1 \
#  | tee data/lsocp1.txt
#
#valgrind --smc-check=all-non-file --leak-check=full \
#  --suppressions=data/valgrind-julia.supp julia data/lsocp2.jl 2>&1 \
#  | tee data/lsocp2.txt
