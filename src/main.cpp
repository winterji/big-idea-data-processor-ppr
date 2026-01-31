#include <OpenCL/opencl.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <complex>
#include <string>
#include <vector>

#include "../include/ReadDexcomData.h"
#include "../include/DataReader.h"

// output files to log into
std::ofstream benchmarkLog("benchmark_logs.txt");
std::ofstream benchmarkCsv("benchmarks.csv");

std::string dataTypeToString(DataType type) {
    switch (type) {
        case DEXCOM: return "DEXCOM";
        case HR:     return "HR";
        case BVP:    return "BVP";
        default:     return "UNKNOWN";
    }
}

template <typename Func>
void measureTime(const std::string& label, Func func, DataType dataType) {
    // 1. Získáme čas před spuštěním (High Resolution Clock)
    auto start = std::chrono::high_resolution_clock::now();

    func();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    double ms = duration.count();

    std::cout << dataTypeToString(dataType) << " >> [BENCHMARK] " << label << ": " << ms << " ms" << std::endl;
    // Zápis do souboru (pokud je otevřený)
    if (benchmarkLog.is_open()) {
        benchmarkLog << dataTypeToString(dataType) << " >> [BENCHMARK] " << label << ": " << ms << " ms" << std::endl;
    }
    if (benchmarkCsv.is_open()) {
        benchmarkCsv << dataTypeToString(dataType) << ", " << label << ", " << ms << std::endl;
    }
}

void control_results(const std::vector<float>& res1, const std::vector<float>& res2, const std::string label, bool printDifferences = false) {
    if (res1.size() != res2.size()) {
        std::cerr << "Different sizes: " << res1.size() << " and " << res2.size() << std::endl;
        return;
    }
    else if (printDifferences) {
        std::cout << "Sizes of " << label << " match: " << res1.size() << std::endl;
    }

    for (size_t i = 0; i < res1.size(); i++) {
        if (std::abs(res1[i] - res2[i]) > 0.0001f) {
            if (printDifferences)
                std::cerr << "Different values on " << i << ": " << res1[i] << " and " << res2[i] << std::endl;
            else {
                std::cerr << "Values " << label << " do not match." << std::endl;
                return;
            }
        }
    }
}

void printData(const std::vector<DexcomRecord>& data) {
    std::cout << "Index\tTimestamp\t\tValue\n";
    std::cout << "-----------------------------------------\n";
    for (const auto& record : data) {
        std::cout << record.mIndex << "\t"
                  << record.mTimestamp << "\t"
                  << record.mValue << '\n';
    }
}

DexcomData readBVPData(int num_files) {
    std::cout << "Reading BVP data..." << std::endl;
    std::vector<ReadDexcomData> data;
    for (auto i = 0; i < num_files; i++) {
        std::string s = std::format("{:03}", i+1);
        const std::string filename = "../data/" + s + "/BVP_" + s + ".csv";
        auto reader = DataReader(filename, BVP);
        if (reader.isOpen()) {
            data.push_back(reader.read());
            std::cout << "File " << filename << " read" << std::endl;
        }
        else {
            std::cerr << "File " << filename <<" failed to open" << std::endl;
        }
    }
    // std::cout << "Size of read data: " << data.at(0).size << std::endl;
    return DataConverter::convertToFlatDataInMilliseconds(data, 5760000, 288, 8);
}

DexcomData readAllHRData() {
    std::cout << "Reading HR data..." << std::endl;
    std::vector<ReadDexcomData> data;
    for (auto i = 0; i < 16; i++) {
        std::string s = std::format("{:03}", i+1);
        const std::string filename = "../data/" + s + "/HR_" + s + ".csv";
        auto reader = DataReader(filename, HR);
        if (reader.isOpen()) {
            data.push_back(reader.read());
            std::cout << "File " << filename << " read" << std::endl;
        }
        else {
            std::cerr << "File " << filename <<" failed to open" << std::endl;
        }
    }
    // std::cout << "Size of read data: " << data.at(0).size << std::endl;
    return DataConverter::convertToFlatDataInSeconds(data, 86400, 288);
}

DexcomData readAllDexcomData() {
    std::vector<ReadDexcomData> data;
    for (auto i = 0; i < 16; i++) {
        std::string s = std::format("{:03}", i+1);
        const std::string filename = "../data/" + s + "/Dexcom_" + s + ".csv";
        auto reader = DataReader(filename, DEXCOM);
        if (reader.isOpen())
            data.emplace_back(reader.read());
        else {
            std::cerr << "File " << filename <<" failed to open" << std::endl;
        }
    }
    auto dexcom = DataConverter::convertToFlatData(data);

    return dexcom;
}

int main() {

    if (benchmarkCsv.is_open()) {
        benchmarkCsv << "Dataset, Algorithm, Time [ms]" << std::endl;
    }

    std::vector<std::string> processData{"dexcom", "hr", "bvp"};

    for (const auto& data : processData) {
        if (data == "dexcom") {
            DexcomData dexcom;
            measureTime("Read of DEXCOM data", [&] () {
                dexcom = readAllDexcomData();
            }, DEXCOM);

            std::cout << "Size of flat data " << dexcom.flat_data.size() << std::endl;
            std::cout << "TimeSlots: " << dexcom.num_time_slots << std::endl;
            std::cout << "Number of patients: " << dexcom.num_patients << std::endl;

            // Sequential
            measureTime("CPU Sequential", [&] () {
                dexcom.processSequential();
            }, DEXCOM);
            // std::cout << "Size of medians: " << dexcom.result_medians_seq.size() << std::endl;
            // std::cout << dexcom.result_medians_seq.at(0) << std::endl;
            // dexcom.result_medians.clear();
            // dexcom.result_medians.reserve(dexcom.num_time_slots * dexcom.num_patients);

            // Parallel
            measureTime("CPU Parallel + Vectorized", [&] () {
                dexcom.processParallelCPU();
            }, DEXCOM);
            // std::cout << "Size of medians: " << dexcom.result_medians_par.size() << std::endl;
            // std::cout << dexcom.result_medians_par.at(0) << std::endl;
            control_results(dexcom.result_medians_seq_per_pat, dexcom.result_medians_par_per_pat, "CPU seq vs CPU par+vect", true);

            // Parallel, no vectorization
            measureTime("CPU Parallel", [&] () {
                dexcom.processParallelCPUNonVectorized();
            }, DEXCOM);
            control_results(dexcom.result_medians_seq_per_pat, dexcom.result_medians_par_nov_vect_per_pat, "CPU seq vs CPU par");

            // GPU
            measureTime("GPU", [&] () {
                dexcom.processGPU(-1, false, 3);
            }, DEXCOM);
            control_results(dexcom.result_medians_seq, dexcom.result_medians_gpu, "CPU seq vs GPU");

            dexcom.exportDebugCSV("dexcom_output.csv");

        }
        // ------------- HR Data --------------
        else if (data == "hr") {
            DexcomData hrData;
            measureTime("Read of HR data", [&] () {
                hrData = readAllHRData();
            }, HR);
            std::cout << "Size of HR data: " << hrData.flat_data.size() << std::endl;
            std::cout << "TimeSlots: " << hrData.num_time_slots << std::endl;
            std::cout << "Number of patients: " << hrData.num_patients << std::endl;

            // Sequential
            measureTime("CPU Sequential", [&] () {
                hrData.processSequential(288);
            }, HR);
            // std::cout << "Size of updated timeslots: " << hrData.updated_timeslots_seq.size() << std::endl;
            // measureTime("CPU Seq new time slots", [&] () {
            //     hrData.processSequentialCPUToNewTimeSlots(288);
            // });

            measureTime("CPU Parallel + Vectorized", [&] () {
                hrData.processParallelCPU(288);
            }, HR);
            // std::cout << "Size of medians through patients: " << hrData.result_medians_par_per_pat.size() << std::endl;
            // std::cout << "Size of medians through only timeslots: " << hrData.result_medians_par.size() << std::endl;

            control_results(hrData.result_medians_seq_per_pat, hrData.result_medians_par_per_pat, "Per patients: Seq vs Par+vect", false);
            control_results(hrData.result_medians_seq, hrData.result_medians_par, "Per timeslots: Seq vs Par+vect", false);
            control_results(hrData.updated_timeslots_seq, hrData.updated_timeslots_par, "Per wanted_timeslots: Seq vs Par+vect", false);

            measureTime("CPU Parallel - NoVect", [&] () {
                hrData.processParallelCPUNonVectorized();
            }, HR);
            control_results(hrData.result_medians_seq_per_pat, hrData.result_medians_par_nov_vect_per_pat, "Seq vs Par-noVect", false);

            measureTime("GPU", [&] () {
                hrData.processGPU(288, false, 3);
            }, HR);
            // control_results(hrData.result_medians_seq_per_pat, hrData.result_medians_gpu_per_pat, "Seq vs GPU", false);
            // control_results(hrData.result_medians_seq, hrData.result_medians_gpu, "Per timeslots: Seq vs GPU", false);
            control_results(hrData.updated_timeslots_seq, hrData.updated_timeslots_gpu, "Per wanted_timeslots: Seq vs GPU", false);

            hrData.exportDebugCSV("hr_output.csv");

        }
        // ------------- BVP Data --------------
        else if (data == "bvp") {
            DexcomData bvpData;

            measureTime("Read of BVP data", [&] () {
                bvpData = readBVPData(16);
            }, BVP);
            std::cout << "Size of BVP data: " << bvpData.flat_data.size() << std::endl;
            std::cout << "TimeSlots: " << bvpData.num_time_slots << std::endl;
            std::cout << "Number of patients: " << bvpData.num_patients << std::endl;

            // Sequential
            measureTime("CPU Sequential", [&] () {
                bvpData.processSequential(288);
            }, BVP);
            std::cout << "Size of updated timeslots: " << bvpData.updated_timeslots_seq.size() << std::endl;

            // Parallel
            measureTime("CPU Parallel + Vectorized", [&] () {
                bvpData.processParallelCPU(288);
            }, BVP);
            // std::cout << "Size of medians through patients: " << hrData.result_medians_par_per_pat.size() << std::endl;
            // std::cout << "Size of medians through only timeslots: " << hrData.result_medians_par.size() << std::endl;

            control_results(bvpData.result_medians_seq_per_pat, bvpData.result_medians_par_per_pat, "Per patients: Seq vs Par+vect", false);
            control_results(bvpData.result_medians_seq, bvpData.result_medians_par, "Per timeslots: Seq vs Par+vect", false);
            control_results(bvpData.updated_timeslots_seq, bvpData.updated_timeslots_par, "Per wanted_timeslots: Seq vs Par+vect", false);

            measureTime("CPU Parallel - NoVect", [&] () {
                bvpData.processParallelCPUNonVectorized();
            }, BVP);
            control_results(bvpData.result_medians_seq_per_pat, bvpData.result_medians_par_nov_vect_per_pat, "Seq vs Par-noVect", false);

            measureTime("GPU - kernel.cl", [&] () {
                bvpData.processGPU(288, false, 3);
            }, BVP);
            // control_results(bvpData.result_medians_seq_per_pat, bvpData.result_medians_gpu_per_pat, "Seq vs GPU", false);
            // control_results(bvpData.result_medians_seq, bvpData.result_medians_gpu, "Per timeslots: Seq vs GPU", false);
            control_results(bvpData.updated_timeslots_par, bvpData.updated_timeslots_gpu, "Per wanted_timeslots: Seq vs GPU", true);

            bvpData.exportDebugCSV("bvp_output.csv");

        }
    }
    benchmarkLog.close();
    benchmarkCsv.close();
    return 0;
};
