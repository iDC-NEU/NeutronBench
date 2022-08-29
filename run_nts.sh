#!/bin/bash
if [ ! -d "build" ]; then
  mkdir build
  cd build && cmake .. && cd ..
fi

cd build && make -j $(nproc) && cd ..
mpiexec -np $1 ./build/nts $2
#mpiexec -np 1 ./nts NTS_cora_data.cfg



