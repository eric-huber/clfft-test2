#include <vector>
#include <iostream>
#include <fstream>
#include <random>
#include <math.h>

// No need to explicitely include the OpenCL headers 
#include <clFFT.h>

const char* _data_file_name = "fft-data.txt";
const char* _fft_file_name  = "fft-forward.txt";
const char* _bak_file_name  = "fft-backward.txt";

typedef std::vector<float> float_v;

size_t              _fft_size = 8192;

float               _mean = 0.5;
float               _std = 0.2;

cl_context          _ctx = NULL;
cl_command_queue    _queue = NULL;
clfftPlanHandle     _plan_forward;
clfftPlanHandle     _plan_backward;
cl_mem              _buf;

void populate_random(size_t size, float_v& buf) {
    std::default_random_engine      generator(std::random_device{}());
    std::normal_distribution<float> distribution(_mean, _std);
    
    srand(time(NULL));

    for(int i = 0; i < size; i++) {
        float number = distribution(generator);
        buf[i] = number;
    }
}

void populate_periodic(size_t size, float_v& buf) {
    for (int i = 0; i < size; ++i) {
        float t = i * .002;
        float amp = sin(M_PI * t);
        amp += sin(2 * M_PI * t);
        amp += sin(3 * M_PI * t); 
        buf[i] = amp; 
    }
}

void populate(size_t size, float_v& buf) {
    //populate_random(size, buf);
    populate_periodic(size, buf);
}

float signal_energy(float_v& input) {
    double energy = 0;
    for (int i = 0; i < input.size(); ++i) {
        energy += pow(input[i], 2);
    }
    return energy;
}

double quant_error_energy(float_v& input, float_v& output) {
    
    double energy = 0;
    for (int i = 0; i < input.size(); ++i) {
        energy += pow(input[i] - output[i], 2);
    }
    return energy;
}

float signal_to_quant_error(float_v& input, float_v& output) {
    return 10.0 * log10( signal_energy(input) / quant_error_energy(input, output) );
}

void write(std::string filename, float_v& buf) {
    std::ofstream ofs;
    ofs.open(filename);
    
    for (int i = 0; i < buf.size(); ++i) {
        ofs << buf[i] << std::endl;
    }
    
    ofs.close();   
}

void write_herm(std::string filename, float_v& buf) {
    std::ofstream ofs;
    ofs.open(filename);
    
    for (int i = 1; i < buf.size() / 2; i+=2) {
        float real = buf[i];
        float imag = buf[i+1];
        float amp = sqrt(pow(real, 2) + pow(imag, 2));
        float phase = atan2(imag, real);
        ofs << amp << ", " << phase << std::endl;
    }
    
    ofs.close();   
}

void cl_init() {
    cl_int                  err;
    cl_platform_id          platform = 0;
    cl_device_id            device = 0;
    cl_context_properties   props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    
    // FFT library realted declarations 
    clfftDim dim = CLFFT_1D;
    size_t clLengths[1] = {_fft_size};

    // Setup OpenCL environment. 
    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    props[1] = (cl_context_properties)platform;
    _ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
    _queue = clCreateCommandQueue(_ctx, device, 0, &err);

    // Setup clFFT. 
    clfftSetupData fftSetup;
    err = clfftInitSetupData(&fftSetup);
    err = clfftSetup(&fftSetup);
    
    // Prepare OpenCL memory objects and place data inside them. 
    _buf = clCreateBuffer(_ctx, CL_MEM_READ_WRITE, _fft_size * 2 * sizeof(float), NULL, &err);

    // Create a default plan for a complex FFT going forward 
    err = clfftCreateDefaultPlan(&_plan_forward, _ctx, dim, clLengths);

    // Set plan parameters. 
    err = clfftSetPlanPrecision(_plan_forward, CLFFT_SINGLE);
    err = clfftSetLayout(_plan_forward, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);
    err = clfftSetResultLocation(_plan_forward, CLFFT_INPLACE);

    // Bake the plan. 
    err = clfftBakePlan(_plan_forward, 1, &_queue, NULL, NULL);
    
    // Create a default plan for a complex FFT going backward 
    err = clfftCreateDefaultPlan(&_plan_backward, _ctx, dim, clLengths);

    // Set plan parameters. 
    err = clfftSetPlanPrecision(_plan_backward, CLFFT_SINGLE);
    err = clfftSetLayout(_plan_backward, CLFFT_HERMITIAN_INTERLEAVED, CLFFT_REAL);
    err = clfftSetResultLocation(_plan_backward, CLFFT_INPLACE);

    // Bake the plan. 
    err = clfftBakePlan(_plan_backward, 1, &_queue, NULL, NULL);
}

void cl_release() {
    cl_int err;
    
    // Release OpenCL memory objects. 
    clReleaseMemObject(_buf);

    // Release the plan. 
    err = clfftDestroyPlan(&_plan_forward);
    err = clfftDestroyPlan(&_plan_backward);

    // Release clFFT library. 
    clfftTeardown();

    // Release OpenCL working objects. 
    clReleaseCommandQueue(_queue);
    clReleaseContext(_ctx);
}

size_t size() {
    //return _fft_size * 2 * sizeof(float);
    return _fft_size * sizeof(float);
}

float basic_fft(float_v& input, float_v& output) {
    cl_int                  err;

    // Make data accessable by OpenCL
    err = clEnqueueWriteBuffer(_queue, _buf, CL_TRUE, 0, size(), &input[0], 0, NULL, NULL);

    // Execute the plan. 
    err = clfftEnqueueTransform(_plan_forward, CLFFT_FORWARD, 1, &_queue, 0, NULL, NULL, &_buf, NULL, NULL);

    // Wait for calculations to be finished. 
    err = clFinish(_queue);

    // Fetch results of calculations. 
    err = clEnqueueReadBuffer(_queue, _buf, CL_TRUE, 0, size(), &output[0], 0, NULL, NULL);
    write_herm(_fft_file_name, output);

    // queue a reverse transform
    err = clfftEnqueueTransform(_plan_backward, CLFFT_BACKWARD, 1, &_queue, 0, NULL, NULL, &_buf, NULL, NULL);

    // Wait for calculations to be finished. 
    err = clFinish(_queue);
    
    // Fetch results of calculations. 
    err = clEnqueueReadBuffer(_queue, _buf, CL_TRUE, 0, size(), &output[0], 0, NULL, NULL);
    write(_bak_file_name, output);

    return signal_to_quant_error(input, output);
}

int main(void)
{
   
    float_v      input(_fft_size);
    float_v      output(_fft_size);
    cl_event     event = NULL;
    int          ret = 0;

    // setup OpenCL & clFFT
    cl_init();

    // populate data
    populate(_fft_size, input);
    write(_data_file_name, input);

    // FFT
    float sqer = basic_fft(input, output);
    std::cout << "SQER:   " << sqer << std::endl;

    // Release resources
    input.empty();
    output.empty();
    cl_release();

    return ret;
}