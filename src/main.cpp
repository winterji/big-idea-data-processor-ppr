#include <OpenCL/opencl.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <complex>
#include <string>
#include <vector>

#include "../include/ReadDexcomData.h"
#include "../include/DataReader.h"
#include "../libs/stringUtils.h"

// output files to log into
std::ofstream benchmarkLog;
std::ofstream benchmarkCsv;

// Configuration structure for CLI args
struct Config {
    bool runDexcom = false;
    bool runHr = false;
    bool runBvp = false;
    
    bool runSeq = false;
    bool runPar = false;
    bool runGpu = false;
    
    int numKernels = 3;
    std::string logFileName = "benchmark_logs.txt";

    bool explicitDatasets = false;
    bool explicitModes = false;
};

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " [options]\n"
              << "Options:\n"
              << "  -d, --datasets <list>    Comma separated list of datasets (dexcom, hr, bvp)\n"
              << "                           Example: -d dexcom,bvp\n"
              << "  -m, --mode <list>        Comma separated list of modes (seq, par, gpu, all)\n"
              << "                           Example: -m seq,gpu\n"
              << "  -k, --kernels <num>      Number of GPU kernels to use (1, 2, or 3). Default: 3\n"
              << "  -o, --output <file>      Output file path for logs. Default: benchmark_logs.txt\n"
              << "  -h, --help               Show this help message\n";
}

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

Config parseCLIArguments(int argc, char* argv[]) {
    Config cfg;
    // --- Command Line Argument Parsing ---
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return cfg;
        } 
        else if (arg == "-d" || arg == "--datasets") {
            if (i + 1 < argc) {
                cfg.explicitDatasets = true;
                std::vector<std::string> ds = split(argv[++i], ',');
                for (const auto& d : ds) {
                    if (d == "dexcom") cfg.runDexcom = true;
                    else if (d == "hr") cfg.runHr = true;
                    else if (d == "bvp") cfg.runBvp = true;
                    else std::cerr << "Unknown dataset: " << d << std::endl;
                }
            }
        } 
        else if (arg == "-m" || arg == "--mode") {
            if (i + 1 < argc) {
                cfg.explicitModes = true;
                std::vector<std::string> ms = split(argv[++i], ',');
                for (const auto& m : ms) {
                    if (m == "seq") cfg.runSeq = true;
                    else if (m == "par") cfg.runPar = true;
                    else if (m == "gpu") cfg.runGpu = true;
                    else if (m == "all") { cfg.runSeq = cfg.runPar = cfg.runGpu = true; }
                    else std::cerr << "Unknown mode: " << m << std::endl;
                }
            }
        } 
        else if (arg == "-k" || arg == "--kernels") {
            if (i + 1 < argc) {
                cfg.numKernels = std::stoi(argv[++i]);
                if (cfg.numKernels < 1 || cfg.numKernels > 3) {
                    std::cerr << "Invalid number of kernels (must be 1-3). Defaulting to 3." << std::endl;
                    cfg.numKernels = 3;
                }
            }
        }
        else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                cfg.logFileName = argv[++i];
            }
        }
    }

    // Defaults if arguments not provided
    if (!cfg.explicitDatasets) {
        cfg.runDexcom = cfg.runHr = cfg.runBvp = true;
    }
    if (!cfg.explicitModes) {
        cfg.runSeq = cfg.runPar = cfg.runGpu = true;
    }
    return cfg;
}

int main(int argc, char* argv[]) {

    Config cfg = parseCLIArguments(argc, argv);

    benchmarkLog.open(cfg.logFileName);
    benchmarkCsv.open("benchmarks.csv");

    if (benchmarkCsv.is_open()) {
        benchmarkCsv << "Dataset, Algorithm, Time [ms]" << std::endl;
    }

    std::vector<std::string> processData;
    if (cfg.runDexcom) processData.push_back("dexcom");
    if (cfg.runHr)     processData.push_back("hr");
    if (cfg.runBvp)    processData.push_back("bvp");

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
            if (cfg.runSeq) {
                measureTime("CPU Sequential", [&] () {
                    dexcom.processSequential();
                }, DEXCOM);
            }

            // Parallel
            if (cfg.runPar) {
                measureTime("CPU Parallel + Vectorized", [&] () {
                    dexcom.processParallelCPU();
                }, DEXCOM);

                // Validation
                if (cfg.runSeq) {
                    control_results(dexcom.result_medians_seq_per_pat, dexcom.result_medians_par_per_pat, "CPU seq vs CPU par+vect", true);
                }

                measureTime("CPU Parallel - NoVect", [&] () {
                    dexcom.processParallelCPUNonVectorized();
                }, DEXCOM);
                
                if (cfg.runSeq) {
                    control_results(dexcom.result_medians_seq_per_pat, dexcom.result_medians_par_nov_vect_per_pat, "CPU seq vs CPU par");
                }
            }

            // GPU
            if (cfg.runGpu) {
                std::string label = "GPU (" + std::to_string(cfg.numKernels) + " kernels)";
                measureTime(label, [&] () {
                    dexcom.processGPU(-1, false, cfg.numKernels);
                }, DEXCOM);
                
                if (cfg.runSeq) {
                    control_results(dexcom.result_medians_seq, dexcom.result_medians_gpu, "CPU seq vs GPU");
                }
            }

            dexcom.exportDebugCSV("dexcom_debug_output.csv");
            dexcom.exportToCSV("dexcom_output.csv");
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
            if (cfg.runSeq) {
                measureTime("CPU Sequential", [&] () {
                    hrData.processSequential(288);
                }, HR);
            }

            // Parallel
            if (cfg.runPar) {
                measureTime("CPU Parallel + Vectorized", [&] () {
                    hrData.processParallelCPU(288);
                }, HR);

                if (cfg.runSeq) {
                    control_results(hrData.result_medians_seq_per_pat, hrData.result_medians_par_per_pat, "Per patients: Seq vs Par+vect", false);
                    control_results(hrData.result_medians_seq, hrData.result_medians_par, "Per timeslots: Seq vs Par+vect", false);
                    control_results(hrData.updated_timeslots_seq, hrData.updated_timeslots_par, "Per wanted_timeslots: Seq vs Par+vect", false);
                }

                measureTime("CPU Parallel - NoVect", [&] () {
                    hrData.processParallelCPUNonVectorized();
                }, HR);
                
                if (cfg.runSeq) {
                    control_results(hrData.result_medians_seq_per_pat, hrData.result_medians_par_nov_vect_per_pat, "Seq vs Par-noVect", false);
                }
            }

            // GPU
            if (cfg.runGpu) {
                std::string label = "GPU (" + std::to_string(cfg.numKernels) + " kernels)";
                measureTime(label, [&] () {
                    hrData.processGPU(288, false, cfg.numKernels);
                }, HR);
                
                if (cfg.runSeq) {
                    control_results(hrData.updated_timeslots_seq, hrData.updated_timeslots_gpu, "Per wanted_timeslots: Seq vs GPU", false);
                }
            }

            hrData.exportDebugCSV("hr_debug_output.csv");
            hrData.exportToCSV("hr_output.csv");

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
            if (cfg.runSeq) {
                measureTime("CPU Sequential", [&] () {
                    bvpData.processSequential(288);
                }, BVP);
                std::cout << "Size of updated timeslots: " << bvpData.updated_timeslots_seq.size() << std::endl;
            }

            // Parallel
            if (cfg.runPar) {
                measureTime("CPU Parallel + Vectorized", [&] () {
                    bvpData.processParallelCPU(288);
                }, BVP);

                if (cfg.runSeq) {
                    control_results(bvpData.result_medians_seq_per_pat, bvpData.result_medians_par_per_pat, "Per patients: Seq vs Par+vect", false);
                    control_results(bvpData.result_medians_seq, bvpData.result_medians_par, "Per timeslots: Seq vs Par+vect", false);
                    control_results(bvpData.updated_timeslots_seq, bvpData.updated_timeslots_par, "Per wanted_timeslots: Seq vs Par+vect", false);
                }
                
                measureTime("CPU Parallel - NoVect", [&] () {
                    bvpData.processParallelCPUNonVectorized();
                }, BVP);
                
                if (cfg.runSeq) {
                    control_results(bvpData.result_medians_seq_per_pat, bvpData.result_medians_par_nov_vect_per_pat, "Seq vs Par-noVect", false);
                }
            }

            // GPU
            if (cfg.runGpu) {
                std::string label = "GPU (" + std::to_string(cfg.numKernels) + " kernels)";
                measureTime(label, [&] () {
                    bvpData.processGPU(288, false, cfg.numKernels);
                }, BVP);

                if (cfg.runPar) {
                    // Validating against Parallel output as Seq might be too slow or not run
                    control_results(bvpData.updated_timeslots_par, bvpData.updated_timeslots_gpu, "Per wanted_timeslots: Par vs GPU", true);
                }
            }

            bvpData.exportDebugCSV("bvp_debug_output.csv");
            bvpData.exportToCSV("bvp_output.csv");
        }
    }
    benchmarkLog.close();
    benchmarkCsv.close();
    return 0;
};
