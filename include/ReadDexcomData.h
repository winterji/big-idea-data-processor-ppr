//
// Created by Jiří Winter on 09.11.2025.
//

#ifndef PPR_SEMESTRALKA_DEXCOMDATA_H
#define PPR_SEMESTRALKA_DEXCOMDATA_H
#include <cstdint>
#include <vector>
#include <algorithm>

constexpr uint32_t TIME_SLOTS_IN_DAY = 288;
constexpr uint32_t SLOT_LENGTH_IN_MINUTES = 5;
constexpr float EMPTY_VALUE = -1.0e30f;
constexpr float EPSILON = 100.0f;

class PatientDataTimeSlot {
public:
    uint16_t time_slot_id;
    std::vector<float> values;  // one value for each day of measuring
    float getMedianValue();
};

class TimeSlot {
public:
    // one time slot contains data for all patients for current time slot
    uint16_t time_slot_id;
    std::vector<PatientDataTimeSlot> patientData;   // one item for patient
    std::vector<float> patientMedianValues;
    float getMedianValue();
};

class DayValues {
public:
    std::vector<TimeSlot> timeSlots;    // size 288 - how many time slots there are
    std::vector<float> timeSlotMedianValues;
};

enum DataType {
    DEXCOM,
    HR,
    BVP
};

class DexcomRecord {
public:
    uint64_t mIndex;
    std::string mTimestamp;
    float mValue;

    DexcomRecord(const uint64_t index, const std::string& timestamp, const float value)
        : mIndex(index), mTimestamp(timestamp), mValue(value) {
        // std::cout << index << timestamp << value << std::endl;
    }
    // DexcomRecord(const std::tuple<uint32_t, std::string, float>& tuple)
    //     : index(std::get<>(tuple))
};

class DexcomData {
public:
    static constexpr uint32_t NUM_TIME_SLOTS = 288;
    static constexpr uint32_t FIXED_NUM_DAYS = 8;

    uint32_t num_time_slots = NUM_TIME_SLOTS;
    uint32_t num_patients = 0;
    uint32_t num_days = FIXED_NUM_DAYS;

    // flat data
    // size = num_time_slots * num_patients * num_days
    // Layout: [TimeSlot][Patient][Day]
    std::vector<float> flat_data;

    // median for [TimeSlot][Patient]
    std::vector<float> result_medians_seq_per_pat;
    std::vector<float> result_medians_par_per_pat;
    std::vector<float> result_medians_par_nov_vect_per_pat;
    std::vector<float> result_medians_gpu_per_pat;

    // median for [TimeSlot]
    std::vector<float> result_medians_seq;
    std::vector<float> result_medians_par;
    std::vector<float> result_medians_par_nov_vect;
    std::vector<float> result_medians_gpu;

    // median for updated time slots [TimeSlot]
    std::vector<float> updated_timeslots_seq;
    std::vector<float> updated_timeslots_par;
    std::vector<float> updated_timeslots_par_non_vect;
    std::vector<float> updated_timeslots_gpu;

    // Pomocná metoda pro přístup k datům (vypočítá index)
    inline float* getPatientDataPtr(uint32_t timeslot, uint32_t patient) {
        size_t index = (size_t)timeslot * num_patients * num_days +
                       (size_t)patient * num_days;

        return &flat_data[index];
    }

    void exportDebugCSV(const std::string& filename);

    void processSequential(int32_t num_wanted_time_slots = -1);
    void processParallelCPU(int32_t num_wanted_time_slots = -1);
    void processParallelCPUNonVectorized(int32_t num_wanted_time_slots = -1);
    void processGPU(int32_t num_wanted_time_slots = -1, bool read_all_outputs = false, int8_t num_kernels_use = 3, const std::string& kernel_file = "../src/kernel.cl");

    void processSequentialCPUToNewTimeSlots(uint32_t numTimeSlots);
};

class ReadDexcomData {
private:
    uint64_t mAlertHigh;
    uint64_t mAlertLow;
    uint64_t mAlertUrgentLow;
    uint64_t mAlertUrgentLowSoon;

public:
    std::vector<DexcomRecord> mData;
    uint32_t size;
};





#endif //PPR_SEMESTRALKA_DEXCOMDATA_H