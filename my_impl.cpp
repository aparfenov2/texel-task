#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cuda.h>
#include <cuda_runtime.h>

// https://stackoverflow.com/questions/49582252/pybind-numpy-access-2d-nd-arrays

namespace py = pybind11;

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

void call_applyFilter(
    const unsigned char *input, 
    unsigned char *output, 
    const unsigned int width, 
    const unsigned int height, 
    const float *kernel, 
    const unsigned int kernelWidth);

// ----------- entry point -------------

void run(py::array _image, py::array _kernel) {

    auto image = _image.request();
    auto kernel = _kernel.request();
    const int num_channels = 3;

    int filterWidth = kernel.shape[0];
    int numRowsImage = image.shape[0];
    int numColsImage = image.shape[1];
    float *h_filter = (float*) kernel.ptr;
    unsigned char *h_image = (unsigned char*) image.ptr;
    
    unsigned char *d_red;
    unsigned char *d_redBlurred;
    float         *d_filter;

    size_t data_size = sizeof(unsigned char) * numRowsImage * numColsImage * num_channels;
    size_t filter_size = sizeof(float) * filterWidth * filterWidth;

    checkCudaErrors(cudaMalloc(&d_redBlurred, data_size ));
    // checkCudaErrors(cudaMemset(d_redBlurred,   127, data_size));

    checkCudaErrors(cudaMalloc(&d_red,   data_size));
    checkCudaErrors(cudaMemcpy(d_red, h_image, data_size, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&d_filter, filter_size));
    checkCudaErrors(cudaMemcpy(d_filter, h_filter, filter_size, cudaMemcpyHostToDevice));

    call_applyFilter(d_red, d_redBlurred, numColsImage, numRowsImage, d_filter, filterWidth);

    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());

    unsigned char *image_out = (unsigned char *) image.ptr;
    checkCudaErrors(cudaMemcpy(image_out, d_redBlurred, data_size, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_redBlurred));
    checkCudaErrors(cudaFree(d_red));
    checkCudaErrors(cudaFree(d_filter));
}


PYBIND11_MODULE(my_impl, m) {
        m.doc() = "the run"; // optional module docstring
        m.def("run", &run, "the run");
}
