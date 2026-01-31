//
// Created by Jiří Winter on 26.12.2025.
//

#include "../include/ReadDexcomData.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

// Include pro OpenCL (Multi-platform)
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

void checkErr(cl_int err, const char* name) {
    if (err != CL_SUCCESS) {
        std::cerr << "[OpenCL Error] " << name << " (" << err << ")" << std::endl;
        exit(1);
    }
}

// !IMPORTANT - KERNEL 2 nastaven pro MAX 16 pacientu
void DexcomData::processGPU(int32_t num_wanted_time_slots, bool read_all_outputs, int8_t num_kernels_use, const std::string& kernel_file) {
    if (num_wanted_time_slots < 0 && num_kernels_use > 2) {
        num_kernels_use = 2;
    }
    cl_int err;

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if(err != CL_SUCCESS) {
        std::cerr << "GPU device not found" << std::endl;
        return;
    }

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkErr(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    checkErr(err, "clCreateCommandQueue");

    // read kernel file
    std::ifstream file(kernel_file);
    if (!file.is_open()) {
        std::cerr << "File " << kernel_file << " can not open" << std::endl;
        exit(1);
    }
    std::string prog(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
    const char* src = prog.c_str();
    size_t src_len = prog.length();

    cl_program program = clCreateProgramWithSource(context, 1, &src, &src_len, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    if (err != CL_SUCCESS) {
        char log[20480];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        std::cerr << "--- Build Log ---\n" << log << "\n-----------------" << std::endl;
        exit(1);
    }

    // celkovy pocet uloh
    size_t total_items = num_time_slots * num_patients;

    // ------------------------
    // Vytvoreni bufferu na GPU

    // input data
    cl_mem d_data = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    flat_data.size() * sizeof(float), flat_data.data(), &err);
    checkErr(err, "CreateBuffer Input Data");

    // medians per patients
    cl_mem d_medians_per_patients_results = clCreateBuffer(context, CL_MEM_WRITE_ONLY, total_items * sizeof(float), NULL, &err);
    checkErr(err, "CreateBuffer medians per patients");

    // medians per time_slots
    cl_mem d_time_slots_results = clCreateBuffer(context, CL_MEM_READ_WRITE, num_time_slots * sizeof(float), NULL, &err);
    checkErr(err, "CreateBuffer medians per time_slots");

    // kernels

    // kernel 1 - median_kernel
    // std::cout << "Calling first kernel - median_kernel" << std::endl;
    cl_kernel kernel = clCreateKernel(program, "median_kernel", &err);
    checkErr(err, "clCreateKernel median_kernel");

    int n_days = (int)num_days;
    int n_total = (int)total_items;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_data);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_medians_per_patients_results);
    clSetKernelArg(kernel, 2, sizeof(int), &n_days);
    clSetKernelArg(kernel, 3, sizeof(int), &n_total);

    size_t global_work_size = total_items;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    checkErr(err, "clEnqueueNDRangeKernel median_kernel");

    if (num_kernels_use == 1 || read_all_outputs) {
        // read output data to vector
        result_medians_gpu_per_pat.resize(total_items);
        err = clEnqueueReadBuffer(queue, d_medians_per_patients_results, CL_TRUE, 0,
                                  total_items * sizeof(float), result_medians_gpu_per_pat.data(),
                                  0, NULL, NULL);
        checkErr(err, "clEnqueueReadBuffer result_medians_gpu_per_pat");
    }

    if (num_kernels_use >= 2) {
        // std::cout << "Calling second kernel - timeslots_kernel" << std::endl;
        // kernel 2 - reduce_patiens_kernel
        cl_kernel timeslots_kernel = clCreateKernel(program, "reduce_patients_kernel", &err);
        checkErr(err, "clCreateKernel reduce_patients_kernel");

        int num_patients_int = static_cast<int>(num_patients);

        clSetKernelArg(timeslots_kernel, 0, sizeof(cl_mem), &d_medians_per_patients_results);
        clSetKernelArg(timeslots_kernel, 1, sizeof(cl_mem), &d_time_slots_results);
        clSetKernelArg(timeslots_kernel, 2, sizeof(int), &num_patients_int);

        size_t global_work_size2 = num_time_slots;

        err = clEnqueueNDRangeKernel(queue, timeslots_kernel, 1, NULL, &global_work_size2, NULL, 0, NULL, NULL);
        checkErr(err, "clEnqueueNDRangeKernel timeslots_kernel");

        // read output data, if only 2 kernels are wanted
        if (num_kernels_use == 2 || read_all_outputs) {
            // read output data to vector
            result_medians_gpu.resize(num_time_slots);
            err = clEnqueueReadBuffer(queue, d_time_slots_results, CL_TRUE, 0,
                                      num_time_slots * sizeof(float), result_medians_gpu.data(),
                                      0, NULL, NULL);
            checkErr(err, "clEnqueueReadBuffer result_medians_gpu");
        }

        // release kernel
        clReleaseKernel(timeslots_kernel);
    }
    //
    //
    // kernel 3 - reduce_slots_kernel
    if (num_kernels_use >= 3 && num_wanted_time_slots > 0) {
        // medians per reduced time_slots
        cl_mem d_wanted_time_slots_results = clCreateBuffer(context, CL_MEM_READ_WRITE, num_wanted_time_slots * sizeof(float), NULL, &err);
        checkErr(err, "CreateBuffer medians per wanted_time_slots");

        // std::cout << "Calling third kernel - reduce_slots_kernel" << std::endl;
        cl_kernel reduce_slots_kernel = clCreateKernel(program, "reduce_slots_kernel", &err);
        checkErr(err, "clCreateKernel reduce_slots_kernel");

        int num_slots = static_cast<int>(num_time_slots);
        int num_wanted_slots = static_cast<int>(num_wanted_time_slots);

        clSetKernelArg(reduce_slots_kernel, 0, sizeof(cl_mem), &d_time_slots_results);
        clSetKernelArg(reduce_slots_kernel, 1, sizeof(cl_mem), &d_wanted_time_slots_results);
        clSetKernelArg(reduce_slots_kernel, 2, sizeof(int), &num_slots);
        clSetKernelArg(reduce_slots_kernel, 3, sizeof(int), &num_wanted_slots);

        size_t global_work_size3 = num_wanted_time_slots;

        err = clEnqueueNDRangeKernel(queue, reduce_slots_kernel, 1, NULL, &global_work_size3, NULL, 0, NULL, NULL);
        checkErr(err, "clEnqueueNDRangeKernel reduce_slots_kernel");

        // read output data, if only 2 kernels are wanted
        if (num_kernels_use == 3) {
            // read output data to vector
            updated_timeslots_gpu.resize(num_wanted_time_slots);
            err = clEnqueueReadBuffer(queue, d_wanted_time_slots_results, CL_TRUE, 0,
                                      num_wanted_time_slots * sizeof(float), updated_timeslots_gpu.data(),
                                      0, NULL, NULL);
            checkErr(err, "clEnqueueReadBuffer updated_timeslots_gpu");
        }

        // release kernel
        clReleaseKernel(reduce_slots_kernel);
        clReleaseMemObject(d_wanted_time_slots_results);
    }


    clReleaseMemObject(d_data);
    clReleaseMemObject(d_medians_per_patients_results);
    clReleaseMemObject(d_time_slots_results);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
