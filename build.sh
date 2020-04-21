#c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` my_impl.cpp -o my_impl`python3-config --extension-suffix`
#exit 0s

# export Torch_DIR="${VIRTUAL_ENV}/lib/python3.7/site-packages/torch/share/cmake/Torch"
mkdir build || true
cd build
cmake .. -DPYTHON_EXECUTABLE=$(which python)
# VERBOSE=1 make -j8
make -j8
cp *.so ..
