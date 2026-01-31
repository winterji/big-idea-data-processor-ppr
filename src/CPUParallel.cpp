//
// Created by Jiří Winter on 22.12.2025.
//

#include "../include/ReadDexcomData.h"
#include <vector>
#include <cmath>
#include <iostream>

#include <arm_neon.h>


#define SORT_VEC(A, B) { \
    float32x4_t min_v = vminq_f32(A, B); \
    float32x4_t max_v = vmaxq_f32(A, B); \
    A = min_v; \
    B = max_v; \
}

inline void transpose_matrix(float32x4_t& r0, float32x4_t& r1, float32x4_t& r2, float32x4_t& r3) {
    float32x4_t t0 = vtrn1q_f32(r0, r1);    // bere všechny sudé indexy
    float32x4_t t1 = vtrn2q_f32(r0, r1);    // bere všechny liché indexy
    float32x4_t t2 = vtrn1q_f32(r2, r3);
    float32x4_t t3 = vtrn2q_f32(r2, r3);
    r0 = vcombine_f32(vget_low_f32(t0), vget_low_f32(t2));
    r1 = vcombine_f32(vget_low_f32(t1), vget_low_f32(t3));
    r2 = vcombine_f32(vget_high_f32(t0), vget_high_f32(t2));
    r3 = vcombine_f32(vget_high_f32(t1), vget_high_f32(t3));
}


// inline void sort_vertical_4(float32x4_t& q0, float32x4_t& q1, float32x4_t& q2, float32x4_t& q3) {
//     SORT_VEC(q0, q1); SORT_VEC(q2, q3);
//     SORT_VEC(q0, q2); SORT_VEC(q1, q3);
//     SORT_VEC(q1, q2);
// }

// // Generováno AI
// // Funkce: Full Sorting Network pro 16 prvků
// // Vstup: 4 registry (q0..q3), každý obsahuje 4 floaty.
// // Výstup: q0..q3 budou obsahovat plně seřazenou posloupnost 0..15.
// // q0 = [0, 1, 2, 3], q1 = [4, 5, 6, 7] ...
inline void bitonic_sort_16(float32x4_t& q0, float32x4_t& q1, float32x4_t& q2, float32x4_t& q3) {

    SORT_VEC(q0, q1); SORT_VEC(q2, q3); SORT_VEC(q0, q2); SORT_VEC(q1, q3);
    SORT_VEC(q0, q3); SORT_VEC(q1, q2);

    transpose_matrix(q0, q1, q2, q3);

    SORT_VEC(q0, q1); SORT_VEC(q2, q3); SORT_VEC(q0, q2); SORT_VEC(q1, q3);
    SORT_VEC(q0, q3); SORT_VEC(q1, q2);

    transpose_matrix(q0, q1, q2, q3);

    // transpose_matrix(q0, q1, q2, q3);

    // sort_vertical_4(q0, q1, q2, q3);
    // transpose_matrix(q0, q1, q2, q3); // Nyní máme seřazené řádky
    //
    // // Merge
    // SORT_VEC(q0, q1); SORT_VEC(q2, q3);
    // transpose_matrix(q0, q1, q2, q3);
    // sort_vertical_4(q0, q1, q2, q3);
    // transpose_matrix(q0, q1, q2, q3);
    //
    // // Final Global Merge
    // SORT_VEC(q0, q2); SORT_VEC(q1, q3);
    // SORT_VEC(q0, q1); SORT_VEC(q2, q3);
    //
    // transpose_matrix(q0, q1, q2, q3);
    // sort_vertical_4(q0, q1, q2, q3);
    // transpose_matrix(q0, q1, q2, q3);
}

// inline float get_median_from_bitonic_sort(float32x4_t& d0, float32x4_t& d1, float32x4_t& d2, float32x4_t& d3) {
//     float32x4_t zero = vdupq_n_f32(0.001f);
//     uint32x4_t counts = vdupq_n_u32(0);

//     counts = vsubq_u32(counts, vcgtq_f32(d0, zero));
//     counts = vsubq_u32(counts, vcgtq_f32(d1, zero));
//     counts = vsubq_u32(counts, vcgtq_f32(d2, zero));
//     counts = vsubq_u32(counts, vcgtq_f32(d3, zero));

//     uint32_t valid_count = vaddvq_u32(counts);

//     float grand_median = 0.0f;

//     if (valid_count > 0) {
//         uint32_t invalid_count = 16 - valid_count;

//         uint32_t median_offset = valid_count / 2;

//         uint32_t target_idx = invalid_count + median_offset;

//         float sorted_buf[16];
//         vst1q_f32(sorted_buf, d0);
//         vst1q_f32(sorted_buf + 4, d1);
//         vst1q_f32(sorted_buf + 8, d2);
//         vst1q_f32(sorted_buf + 12, d3);

//         grand_median = sorted_buf[target_idx];

//         if (valid_count % 2 == 0) {
//             grand_median = (grand_median + sorted_buf[target_idx - 1]) / 2.0f;
//         }
//     }
//     return grand_median;
// }

void DexcomData::processParallelCPU(int32_t num_wanted_time_slots) {
    // Zarovnání na čtveřice
    const uint32_t aligned_patients = (num_patients / 4) * 4;
    const uint32_t remainder = num_patients % 4;
    const uint32_t padding_patients = (remainder == 0) ? 0 : (4 - remainder);

    for (auto i = 0; i < padding_patients; i++) {
        for (auto j = 0; j < num_time_slots; j++) {
            flat_data.push_back(-1);
        }
    }
    const uint32_t patients_with_padding = aligned_patients + padding_patients;

    #pragma omp parallel for
    {
        for (uint32_t ts = 0; ts < num_time_slots; ++ts) {
            // pro median napric pacienty
            std::vector<float> grand_median_buffer;
            grand_median_buffer.reserve(num_patients);
            // 4 pacienty najednou
            for (uint32_t p = 0; p < patients_with_padding; p += 4) {

                float* ptr = getPatientDataPtr(ts, p);

                float32x4_t d0 = vld1q_f32(ptr);
                float32x4_t d1 = vld1q_f32(ptr + 1*8);
                float32x4_t d2 = vld1q_f32(ptr + 2*8);
                float32x4_t d3 = vld1q_f32(ptr + 3*8);
                transpose_matrix(d0, d1, d2, d3);

                float32x4_t d4 = vld1q_f32(ptr + 4);
                float32x4_t d5 = vld1q_f32(ptr + 1*8 + 4);
                float32x4_t d6 = vld1q_f32(ptr + 2*8 + 4);
                float32x4_t d7 = vld1q_f32(ptr + 3*8 + 4);
                transpose_matrix(d4, d5, d6, d7);

                // Generováno AI
                SORT_VEC(d0, d1); SORT_VEC(d2, d3); SORT_VEC(d4, d5); SORT_VEC(d6, d7);
                SORT_VEC(d0, d2); SORT_VEC(d1, d3); SORT_VEC(d4, d6); SORT_VEC(d5, d7);
                SORT_VEC(d1, d2); SORT_VEC(d5, d6); SORT_VEC(d0, d4); SORT_VEC(d3, d7);
                SORT_VEC(d1, d5); SORT_VEC(d2, d6);
                SORT_VEC(d1, d4); SORT_VEC(d3, d6);
                SORT_VEC(d2, d4); SORT_VEC(d3, d5);
                SORT_VEC(d3, d4); SORT_VEC(d1, d2); SORT_VEC(d5, d6);

                float32x4_t limit = vdupq_n_f32(EMPTY_VALUE + EPSILON);
                uint32x4_t counts = vdupq_n_u32(0);

                // Sčítáme masky (True = -1)
                counts = vsubq_u32(counts, vcgtq_f32(d0, limit));
                counts = vsubq_u32(counts, vcgtq_f32(d1, limit));
                counts = vsubq_u32(counts, vcgtq_f32(d2, limit));
                counts = vsubq_u32(counts, vcgtq_f32(d3, limit));
                counts = vsubq_u32(counts, vcgtq_f32(d4, limit));
                counts = vsubq_u32(counts, vcgtq_f32(d5, limit));
                counts = vsubq_u32(counts, vcgtq_f32(d6, limit));
                counts = vsubq_u32(counts, vcgtq_f32(d7, limit));

                uint32_t cnt[4];
                vst1q_u32(cnt, counts);

                float sorted_cols[32];
                vst1q_f32(sorted_cols + 0, d0);
                vst1q_f32(sorted_cols + 4, d1);
                vst1q_f32(sorted_cols + 8, d2);
                vst1q_f32(sorted_cols + 12, d3);
                vst1q_f32(sorted_cols + 16, d4);
                vst1q_f32(sorted_cols + 20, d5);
                vst1q_f32(sorted_cols + 24, d6);
                vst1q_f32(sorted_cols + 28, d7);

                for(int k=0; k<4; ++k) {
                    int n = cnt[k];
                    float median = EMPTY_VALUE;
                    if(n > 0) {

                        int start = 8 - n;
                        int mid = n / 2;

                        int idx1 = (start + mid) * 4 + k;
                        median = sorted_cols[idx1];

                        if(n % 2 == 0) {
                            int idx2 = (start + mid - 1) * 4 + k;
                            median = (sorted_cols[idx2] + median) / 2.0f;
                        }
                        grand_median_buffer.push_back(median);
                    }
                    result_medians_par_per_pat[(ts * num_patients) + p + k] = median;
                    // if (ts < 1)
                    //     std::cout << "Pushing median per patient on index " << ((ts * num_patients) + p + k) << " : " << median << std::endl;
                }
            }
            // TODO tady by melo probehnout zpracovani pacientu, ktere se nevesli do nasobku 4
            // // Zpracování pacientů, kteří se nevešli do čtveřice (zbytek dělení 4)
            // for (uint32_t p = aligned_patients; p < num_patients; ++p) {
            //     float* ptr = getPatientDataPtr(ts, p);
            //     std::vector<float> vals;
            //     for(int d=0; d<8; ++d) {
            //         if (ptr[d] > EMPTY_SLOT_VALUE + 1000.0f) vals.push_back(ptr[d]);
            //     }
            //     float median = EMPTY_SLOT_VALUE;
            //     if (!vals.empty()) {
            //         size_t n = vals.size()/2;
            //         std::nth_element(vals.begin(), vals.begin()+n, vals.end());
            //         median = vals[n];
            //         if (vals.size()%2==0) {
            //             auto max_it = std::max_element(vals.begin(), vals.begin()+n);
            //             median = (*max_it + median)/2.0f;
            //         }
            //         grand_median_buffer.push_back(median);
            //     }
            //     result_medians_par_per_pat[(size_t)(ts * num_patients) + p] = median;
            // }


            // const float* ptr = &result_medians_par_per_pat[ts*num_patients];
            // // median napric pacienty
            // float32x4_t d0 = vld1q_f32(ptr);
            // float32x4_t d1 = vld1q_f32(ptr + 4);
            // float32x4_t d2 = vld1q_f32(ptr + 8);
            // float32x4_t d3 = vld1q_f32(ptr + 12);
            //
            // bitonic_sort_16(d0, d1, d2, d3);

            // float grand_median = get_median_from_bitonic_sort(d0, d1, d2, d3);
            // if (ts < 1)
            //     std::cout << "Median per time slot " << ts << ": " << grand_median << std::endl;

            // median napric pacienty sekvencne pomoci std - sekvencne
            // pro HR data samotne prodlouzilo cas behu na dvojnasobek -> y 80ms na 170ms
            // const float* ptr = &result_medians_par_per_pat[ts*num_patients];
            // std::vector<float> patient_values(ptr, ptr + num_patients);
            // size_t n = patient_values.size() / 2;
            // std::nth_element(patient_values.begin(), patient_values.begin() + n, patient_values.end());
            // float grand_median;
            //
            // if (patient_values.size() % 2 != 0) {
            //     // Lichý počet prvků
            //     grand_median = patient_values[n];
            // } else {
            //     // Sudý počet prvků - medián je průměr dvou prostředních
            //     // Musíme najít i ten druhý prvek (max z prvků před n)
            //     auto it = std::max_element(patient_values.begin(), patient_values.begin() + n);
            //     grand_median = (*it + patient_values[n]) / 2.0f;
            // }

            float gm = EMPTY_VALUE;
            if (!grand_median_buffer.empty()) {
                size_t n = grand_median_buffer.size() / 2;
                std::nth_element(grand_median_buffer.begin(), grand_median_buffer.begin() + n, grand_median_buffer.end());
                gm = grand_median_buffer[n];

                if (grand_median_buffer.size() % 2 == 0) {
                    auto max_it = std::max_element(grand_median_buffer.begin(), grand_median_buffer.begin() + n);
                    gm = (*max_it + gm) / 2.0f;
                }
            }

            // Uložení výsledku
            result_medians_par[ts] = gm;
        }
    }
    // if (num_wanted_time_slots > 0) {
    //     // median napric time slots -> kdyz chci zmensit mnozstvi timeslots behem dne
    //     auto oldInNew = num_time_slots / num_wanted_time_slots;
    //     auto capacity = std::ceil(oldInNew/16.0)*16;
    //     #pragma omp parallel for
    //     for (auto i = 0; i < num_time_slots; i += oldInNew) {
    //         std::vector<float> timeslot_medians;
    //         timeslot_medians.reserve(capacity/16);
    //         // std::cout << "Pushing to timeslot_medans: ";
    //         for (auto j = 0; j < capacity; j += 16) {
    //             // std::cout << "(" << result_medians_par[i + j] << ", " << result_medians_par[i + j +1] << ", " << result_medians_par[i + j +2] << ", " << result_medians_par[i + j +3] << ", " << result_medians_par[i + j +4] << ", " << result_medians_par[i + j +5] << ", " << result_medians_par[i + j +6] << ", " << result_medians_par[i + j +7] << ", " << result_medians_par[i + j +8] << ", " << result_medians_par[i + j +9] << ", " << result_medians_par[i + j +10] << ", " << result_medians_par[i + j +11] << ", " << result_medians_par[i + j +12] << ", " << result_medians_par[i + j +13] << ", " << result_medians_par[i + j +14] << ", " << result_medians_par[i + j +15] << ") -> ";
    //             const float* ptr = &result_medians_par[i + j];
    //             float32x4_t d0 = vld1q_f32(ptr);
    //             float32x4_t d1 = vdupq_n_f32(0.0f);
    //             if (result_medians_par.size() > i + j + 4)
    //                 d1 = vld1q_f32(ptr + 4);
    //             float32x4_t d2 = vdupq_n_f32(0.0f);
    //             if (result_medians_par.size() > i + j + 8)
    //                 d2 = vld1q_f32(ptr + 8);
    //             float32x4_t d3 = vdupq_n_f32(0.0f);
    //             if (result_medians_par.size() > i + j + 8)
    //                 d3 = vld1q_f32(ptr + 12);
    //
    //             bitonic_sort_16(d0, d1, d2, d3);
    //             float grand_median = get_median_from_bitonic_sort(d0, d1, d2, d3);
    //             // std::cout << grand_median << ", ";
    //             timeslot_medians.push_back(grand_median);
    //         }
    //         // std::cout << std::endl;
    //         std::nth_element(timeslot_medians.begin(), timeslot_medians.begin() + timeslot_medians.size()/2, timeslot_medians.end());
    //         updated_timeslots_par[i/oldInNew] = timeslot_medians[timeslot_medians.size()/2];
    //     }
    // }

    if (num_wanted_time_slots > 0 && num_time_slots > 0) {
        auto start = std::chrono::high_resolution_clock::now();
        if (updated_timeslots_par.size() != num_wanted_time_slots) {
            updated_timeslots_par.resize(num_wanted_time_slots);
        }

        // pro HR data tato cast trva 2.20817 ms
        #pragma omp parallel
        {
            std::vector<float> local_buffer;
            local_buffer.reserve((num_time_slots / num_wanted_time_slots) + 2);

            #pragma omp for schedule(static)
            for (int i = 0; i < num_wanted_time_slots; ++i) {

                size_t start_idx = (static_cast<size_t>(i) * num_time_slots) / num_wanted_time_slots;
                size_t end_idx = (static_cast<size_t>(i + 1) * num_time_slots) / num_wanted_time_slots;

                // pokud v poslednim intervalu by bylo mene hodnot
                if (end_idx > num_time_slots) end_idx = num_time_slots;

                size_t count = end_idx - start_idx;

                if (count <= 0) {
                    updated_timeslots_par[i] = (start_idx < num_time_slots) ? result_medians_par[start_idx] : EMPTY_VALUE;
                    continue;
                }

                local_buffer.clear();
                for (size_t k = start_idx; k < end_idx; ++k) {
                    float val = result_medians_par[k];
                    if (val > EMPTY_VALUE + EPSILON) {
                        local_buffer.push_back(val);
                    }
                }


                // if size if even, median is average of the two next t
                // timhle prodlouzeno o 0.4 ms
                float median = EMPTY_VALUE;
                if (!local_buffer.empty()) {
                    auto n = local_buffer.size() / 2;
                    std::nth_element(local_buffer.begin(), local_buffer.begin() + n, local_buffer.end());
                    median = local_buffer[n];

                    if (n % 2 == 0) {
                        auto max_it = std::max_element(local_buffer.begin(), local_buffer.begin() + n);
                        median = (*max_it + median) / 2.0f;
                    }
                }
                updated_timeslots_par[i] = median;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Median napric timeslots in parallel trval: " << ": " << duration.count() << " ms" << std::endl;
    }

}

float calculateMedianPar(std::vector<float>& data) {
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

void DexcomData::processParallelCPUNonVectorized(int32_t num_wanted_time_slots) {
    #pragma omp parallel for
    for (auto i = 0; i < num_time_slots; i++) {
        for (auto j = 0; j < num_patients; j++) {
            std::vector<float> vectorData;
            vectorData.reserve(num_days);
            auto data = getPatientDataPtr(i, j);
            for (auto k = 0; k < num_days; k++) {
                vectorData.push_back(*(data+k));
            }
            auto median = calculateMedianPar(vectorData);
            result_medians_par_nov_vect_per_pat[i * num_patients + j] = median;
            // std::cout << "Median on index " << result_medians_seq.size() << ": " << median << std::endl;
        }
    }
}
