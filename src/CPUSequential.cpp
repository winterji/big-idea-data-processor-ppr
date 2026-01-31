// //
// // Created by Jiří Winter on 13.12.2025.
// //

#include <dispatch/dispatch.h>
#include <algorithm>
#include <vector>
#include <iostream>

#include "../include/ReadDexcomData.h"

inline float calculateMedian(std::vector<float>& data) {
    if (data.empty()) return EMPTY_VALUE;

    std::vector<float> valid_data;
    valid_data.reserve(data.size());

    for (float val : data) {
        if (val > EMPTY_VALUE + EPSILON) {
            valid_data.push_back(val);
        }
    }

    if (valid_data.empty()) {
        return EMPTY_VALUE;
    }

    size_t n = valid_data.size() / 2;
    std::nth_element(valid_data.begin(), valid_data.begin() + n, valid_data.end());

    float median = valid_data[n];

    if (valid_data.size() % 2 == 0) {
        auto max_it = std::max_element(valid_data.begin(), valid_data.begin() + n);
        median = (*max_it + median) / 2.0f;
    }

    return median;
}

void DexcomData::processSequentialCPUToNewTimeSlots(uint32_t newNumTimeSlots) {
    if (newNumTimeSlots >= num_time_slots / 2) {
        std::cerr << "Can not calculate median smaller time slots then recorded.";
        return;
    }
    auto oldSlotsInNew = num_time_slots / newNumTimeSlots;

    #pragma clang loop vectorize(disable) interleave(disable)
    for (auto i = 0; i < num_patients; i++) {
        for (auto j = 0; j < num_time_slots; j += oldSlotsInNew) {
            std::vector<float> vectorData;
            vectorData.reserve(oldSlotsInNew);
            for (auto k = 0; k < oldSlotsInNew; k++) {
                vectorData.push_back(result_medians_seq[j * num_patients + k + i]);
            }
            auto median = calculateMedian(vectorData);
            updated_timeslots_seq.push_back(median);
        }
    }

}


void DexcomData::processSequential(int32_t num_wanted_time_slots) {
#pragma clang loop vectorize(enable) interleave(disable)
    for (auto i = 0; i < num_time_slots; i++) {
        std::vector<float> patients_medians;
        patients_medians.reserve(num_patients);
        for (auto j = 0; j < num_patients; j++) {
            std::vector<float> vectorData;
            vectorData.reserve(num_days);
            auto data = getPatientDataPtr(i, j);
            // std::cout << "Pushed from flat data: ";
            for (auto k = 0; k < num_days; k++) {
                // std::cout << *(data+k) << ", ";
                vectorData.push_back(*(data+k));
            }
            // std::cout << std::endl;
            auto median = calculateMedian(vectorData);
            // if (i < 1)
            //     std::cout << "Median per patient on timeslot " << i << ":" << median << std::endl;
            result_medians_seq_per_pat.push_back(median);
            patients_medians.push_back(median);
            // std::cout << "Median on index " << result_medians_seq.size() << ": " << median << std::endl;
        }
        auto median = calculateMedian(patients_medians);
        // if (i < 1)
        //     std::cout << "Median per timeslot " << i << ":" << median << std::endl;
        result_medians_seq.push_back(median);
    }
    if (num_wanted_time_slots > 0) {
        // std::cout << "Size of updated: " << updated_timeslots_seq.size() << std::endl;
        updated_timeslots_seq.clear();
        updated_timeslots_seq.reserve(num_wanted_time_slots);

        std::vector<float> vectorData;
        vectorData.reserve((num_time_slots / num_wanted_time_slots) + 2);

        for (int i = 0; i < num_wanted_time_slots; ++i) {
            size_t start_idx = (static_cast<size_t>(i) * num_time_slots) / num_wanted_time_slots;
            size_t end_idx = (static_cast<size_t>(i + 1) * num_time_slots) / num_wanted_time_slots;

            if (end_idx > num_time_slots) end_idx = num_time_slots;

            if (start_idx >= end_idx) {
                updated_timeslots_seq.push_back(EMPTY_VALUE);
                continue;
            }

            vectorData.clear();
            for (size_t k = start_idx; k < end_idx; ++k) {
                vectorData.push_back(result_medians_seq[k]);
            }

            auto median = calculateMedian(vectorData);

            updated_timeslots_seq.push_back(median);
        }
    }

}