#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <torch/extension.h>

// https://stackoverflow.com/questions/49582252/pybind-numpy-access-2d-nd-arrays

namespace py = pybind11;

//def run(image: np.ndarray, kernel_size: int, sigma: float):

py::handle run(py::handle _image, int kernel_size, double sigma) {

    auto image = torch::utils::tensor_from_numpy(_image.ptr());

    auto x_cord = torch::arange(kernel_size);
    auto x_grid = x_cord.repeat(kernel_size).view({kernel_size, kernel_size});
    auto y_grid = x_grid.t();
    auto xy_grid = torch::stack({x_grid, y_grid}, -1);

    auto mean = double(kernel_size - 1) / 2.0d;
    auto variance = sigma * sigma;

    auto kernel1 = 1.0d / (2.0d * M_PI * variance);
    auto kernel = torch::pow(xy_grid - mean, 2);
    kernel = torch::sum(kernel, -1);
    kernel = torch::exp(-kernel / (2.0d * variance));
    kernel = kernel * kernel1;
    kernel = kernel / torch::sum(kernel);

    auto channels = image.sizes()[2];

    kernel = kernel.view({1, 1, kernel_size, kernel_size});
    kernel = kernel.repeat({channels, 1, 1, 1});
    kernel = kernel.cuda();
    // kernel.requires_grad(false);
    // f = nn.Conv2d(in_channels=channels, out_channels=channels,
    //               kernel_size=kernel_size, groups=channels, bias=False)
    auto opts = torch::nn::Conv2dOptions(channels, channels, kernel_size).groups(channels).bias(false);
    auto f = torch::nn::Conv2d(opts);
    f->weight = kernel;
    f->weight.options().requires_grad(false);

	image = image.to(torch::kFloat);
    image = image.permute({2, 1, 0}); // HWC -> CWH
    image = image.unsqueeze(0);
    image = image.cuda();

    auto ret = f(image);
    ret = ret.cpu();
    ret = ret.squeeze();

    auto ret_np  = torch::utils::tensor_to_numpy(ret);

    return ret_np; //result;
}


PYBIND11_MODULE(my_impl, m) {
        m.doc() = "the run"; // optional module docstring
        m.def("run", &run, "the run");
}
