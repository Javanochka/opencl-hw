#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>

size_t const BLOCK_SIZE = 256;

size_t div_up(size_t a, size_t b) {
    return (a + b - 1) / b;
}

size_t round_up(size_t n, size_t block_size) {
    return div_up(n, block_size) * block_size;
}

void partial_copy(std::vector<double> const &from, std::vector<double> &to, cl::Context &context, cl::Program &program,
                  cl::CommandQueue &queue) {
    cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(double) * from.size());
    cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(double) * to.size());

    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(double) * from.size(), &from[0]);

    size_t rounded_size = round_up(from.size(), BLOCK_SIZE);

    cl::Kernel kernel(program, "block_copy");

    cl::KernelFunctor convolve_functor(kernel, queue, cl::NullRange, cl::NDRange(rounded_size), cl::NDRange(BLOCK_SIZE));
    convolve_functor(dev_input, dev_output, static_cast<unsigned int>(from.size()), static_cast<unsigned int>(to.size()));

    to[0] = 0.0;
    queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(double) * to.size(), &to[0]);
}

void block_add(std::vector<double> const &from, std::vector<double> &to, cl::Context &context, cl::Program &program,
               cl::CommandQueue &queue) {
    cl::Buffer dev_input_partial(context, CL_MEM_READ_ONLY, sizeof(double) * from.size());
    cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(double) * to.size());
    cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(double) * to.size());

    queue.enqueueWriteBuffer(dev_input_partial, CL_TRUE, 0, sizeof(double) * from.size(), &from[0]);
    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(double) * to.size(), &to[0]);

    size_t rounded_size = round_up(to.size(), BLOCK_SIZE);

    cl::Kernel kernel(program, "block_add");

    cl::KernelFunctor convolve_functor(kernel, queue, cl::NullRange, cl::NDRange(rounded_size), cl::NDRange(BLOCK_SIZE));
    convolve_functor(dev_input_partial, dev_input, dev_output, static_cast<unsigned int>(to.size()));

    queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(double) * to.size(), &to[0]);
}

void prefix_sum(std::vector<double> &res, cl::Context &context, cl::Program &program, cl::CommandQueue &queue) {
    cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(double) * res.size());
    cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(double) * res.size());

    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(double) * res.size(), &res[0]);

    size_t rounded_size = round_up(res.size(), BLOCK_SIZE);

    cl::Kernel kernel(program, "prefix_sum_group");
    cl::KernelFunctor convolve_functor(kernel, queue, cl::NullRange, cl::NDRange(rounded_size), cl::NDRange(BLOCK_SIZE));

    convolve_functor(dev_input, dev_output, cl::__local(sizeof(double) * BLOCK_SIZE), cl::__local(sizeof(double) * BLOCK_SIZE),
                     static_cast<unsigned int>(res.size()));

    queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(double) * res.size(), &res[0]);

    if (res.size() > BLOCK_SIZE) {
        std::vector<double> sums(div_up(res.size(), BLOCK_SIZE));
        partial_copy(res, sums, context, program, queue);
        prefix_sum(sums, context, program, queue);
        block_add(sums, res, context, program, queue);
    }
}

int main()
{
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {
        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_CPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        // load opencl source
        std::ifstream cl_file("prefix_sum.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
                                                      cl_string.length() + 1));

        // create program
        cl::Program program(context, source);

        // compile opencl source
        try
        {
            program.build(devices);
        }
        catch (cl::Error const & e)
        {
            std::string log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
            std::cout << log_str;
            return 0;
        }

        std::ifstream fin("input.txt");
        std::ofstream fout("output.txt");

        size_t test_n = 0;

        fin >> test_n;

        size_t rounded_n = round_up(test_n, BLOCK_SIZE);

        std::vector<double> a(rounded_n, 0);
        for (size_t i = 0; i < test_n; ++i) {
            fin >> a[i];
        }

        prefix_sum(a, context, program, queue);

        for (size_t i = 0; i < test_n; ++i) {
            fout << a[i] << " ";
        }
        fin.close();
        fout.close();
    }
    catch (cl::Error const & e)
    {
        std::cout << "!" << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}