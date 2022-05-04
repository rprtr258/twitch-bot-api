wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcpu.zip
export LIBTORCH=$(pwd)
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
