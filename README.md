# Parallel Processing of BIG IDEAs Lab Glycemic Variability and Wearable Device Data in C++

**Author:** Jiří Winter  
**Date:** January 2026

![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![OpenCL](https://img.shields.io/badge/OpenCL-929292?style=for-the-badge&logo=opencl&logoColor=white)
![OpenMP](https://img.shields.io/badge/OpenMP-blue?style=for-the-badge)
![CMake](https://img.shields.io/badge/CMake-%23008FBA.svg?style=for-the-badge&logo=cmake&logoColor=white)

## 📋 Overview

The goal of this work is the effective processing of large volumes of medical data from glucose sensors and wearable devices. The application handles data with varying sampling frequencies, ranging from **5 minutes down to 15 ms**.

Key functions include:
* Calculating **medians** of measured values over time.
* Performing **data reduction** (downsampling).
* Utilizing **CPU parallelization** (OpenMP, SIMD).
* Utilizing **GPU parallelization** (OpenCL).

## 🩺 Data Sources

The application is designed to process three specific types of medical data:

| Data Type | Description | Sampling Frequency |
| :--- | :--- | :--- |
| **Dexcom** | Continuous Glucose Monitoring (CGM) levels | ~5 minutes |
| **HR** | Heart Rate data | ~1 second |
| **BVP** | Blood Volume Pulse (raw PPG signal) | ~15 ms |

## 🚀 Features

* **High-Performance Processing:** Optimized for large datasets using C++23.
* **Parallel Computing:**
    * **CPU:** Multi-threading via OpenMP and manual vectorization using SIMD instructions (NEON).
    * **GPU:** Accelerated processing using OpenCL kernels.
* **Flexible Pipeline:** Configurable via command-line arguments to switch between sequential, parallel CPU, and GPU modes.

## 📂 Project Structure

The project is organized into modules based on responsibility. The input data must be placed in a `data/` folder in the project root.

```text
.
├── data/                       # Input CSV files (Must be added manually)
│   ├── 001/
│   │   ├── DEXCOM_001.csv      # Glucose data (5 min)
│   │   ├── HR_001.csv          # Heart Rate (1 sec)
│   │   └── BVP_001.csv         # Blood Volume Pulse (15 ms)
│   ├── 002/
│   └── ...
├── include/                    # Header files
│   ├── DataReader.h            # Data loading and conversion declarations
│   └── ReadDexcomData.h        # Data structures for Dexcom/HR/BVP
├── src/                        # Source code
│   ├── CPUParallel.cpp         # OpenMP and SIMD implementation
│   ├── CPUSequential.cpp       # Reference sequential implementation
│   ├── GPUParallel.cpp         # OpenCL host code (buffer management)
│   ├── DataReader.cpp          # CSV parsing and timestamp normalization
│   ├── kernel.cl               # OpenCL kernels (run on GPU)
│   └── main.cpp                # Entry point and benchmarking
├── CMakeLists.txt              # CMake build configuration
└── README.md
```

## 🛠️ Build Instructions

### Prerequisites
* C++ Compiler (supporting C++23)
* CMake (version 3.10 or higher)
* OpenCL headers and libraries
* OpenMP support

### Compilation
```bash
mkdir build
cd build
cmake ..
make
```

## 📜 License

This project was created as a semester work for the **KIV/PPR** (Parallel Programming) course at the **University of West Bohemia** (Západočeská univerzita v Plzni).

The source code is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

**Note:** The medical datasets used for benchmarking are not included in this repository, but are available at [physionet.org](https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/)
