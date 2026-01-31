//
// Created by Jiří Winter on 09.11.2025.
//
#include <iostream>
#include <fstream>
#include "../include/DataReader.h"

#include <algorithm>
#include <charconv>
#include <unordered_map>

#include "../libs/stringUtils.h"

namespace detail {
    inline std::optional<float> median_inplace(std::vector<float>& v) {
        if (v.empty()) return std::nullopt;
        const auto n = v.size();
        std::ranges::nth_element(v, v.begin() + n / 2);
        float mid = v[n / 2];
        if (n % 2 == 1) return mid;
        // need the lower middle as well
        std::ranges::nth_element(v, v.begin() + (n / 2 - 1));
        return (mid + v[n / 2 - 1]) * 0.5f;
    }

    inline float median_copy(std::vector<float> v, float fallback = 0.0f) {
        if (v.empty()) return fallback;
        auto m = median_inplace(v);
        return m.value_or(fallback);
    }
}

float PatientDataTimeSlot::getMedianValue() {
    return detail::median_copy(values, 0.0f);
}

float TimeSlot::getMedianValue() {
    // Flatten all patient values for this slot and compute median.
    std::vector<float> all;
    // Reserve a rough size to reduce reallocations (optional).
    size_t total = 0;
    for (const auto& p : patientData) total += p.values.size();
    all.reserve(total);

    for (const auto& p : patientData) {
        all.insert(all.end(), p.values.begin(), p.values.end());
    }
    return detail::median_copy(all, 0.0f);
}

DataReader::DataReader(const std::string &filename, DataType dataType)
    : mFilename(filename), mFile(std::make_unique<std::ifstream>(mFilename)), mDataType(dataType) {
    if (!mFile->is_open()) {
        std::cerr << "Soubor se nepodařilo otevřít!\n" << std::endl;
    }
}

DataReader::~DataReader() {
    if (mFile && mFile->is_open()) {
        mFile->close();
    }
}

std::tuple<uint32_t, std::string, float> DataReader::parseLine(const std::string &line) {
    std::vector<std::string> tokens = split(line, ',' );
    std::string timestamp, eventType;
    float value;
    uint32_t index;

    switch (mDataType) {
        case DEXCOM:
            index = std::stoi(tokens[0]);
            timestamp = tokens[1];
            eventType = tokens[2];
            if (eventType != "EGV")
                return std::make_tuple(index, timestamp, -1);
            value = std::stof(tokens[7]);

            return std::make_tuple(index, timestamp, value);

        case HR:
        case BVP:
            timestamp = tokens[0];
            value = std::stof(tokens[1]);
            // std::cout << value << std::endl;
            return std::make_tuple(0, timestamp, value);

        default:
            std::cerr << "Unknown data type: " << mDataType << std::endl;

    }

}

void DexcomData::exportDebugCSV(const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Nelze otevřít soubor pro zápis: " << filename << std::endl;
        return;
    }

    // 1. Hlavička CSV
    file << "Index, PER PAT: Seq, Par, GPU,   |   PER SLOT: Seq, Par, GPU,   |   REDUCE: SEQ, PAR, GPU\n";

    file << std::fixed << std::setprecision(6);

    size_t max_size = result_medians_seq_per_pat.size();
    if (result_medians_par_per_pat.size() > max_size) max_size = result_medians_par_per_pat.size();
    if (result_medians_gpu_per_pat.size() > max_size) max_size = result_medians_gpu_per_pat.size();

    for (size_t i = 0; i < max_size; ++i) {
        float val_seq = (i < result_medians_seq_per_pat.size()) ? result_medians_seq_per_pat[i] : -999.0f;
        float val_par = (i < result_medians_par_per_pat.size()) ? result_medians_par_per_pat[i] : -999.0f;
        float val_gpu = (i < result_medians_gpu_per_pat.size()) ? result_medians_gpu_per_pat[i] : -999.0f;

        float val_seq_slot = (i < result_medians_seq.size()) ? result_medians_seq[i] : -999.0f;
        float val_par_slot = (i < result_medians_par.size()) ? result_medians_par[i] : -999.0f;
        float val_gpu_slot = (i < result_medians_gpu.size()) ? result_medians_gpu[i] : -999.0f;

        float val_seq_reduce = (i < updated_timeslots_seq.size()) ? updated_timeslots_seq[i] : -999.0f;
        float val_par_reduce = (i < updated_timeslots_par.size()) ? updated_timeslots_par[i] : -999.0f;
        float val_gpu_reduce = (i < updated_timeslots_gpu.size()) ? updated_timeslots_gpu[i] : -999.0f;

        float diff_sp = std::abs(val_seq - val_par);
        float diff_sg = std::abs(val_seq - val_gpu);

        file << i << ","
            << val_seq << ","
            << val_par << ","
            << val_gpu << ","
            << "  |  "
            << val_seq_slot << ","
            << val_par_slot << ","
            << val_gpu_slot << ","
            << "  |  "
            << ((val_seq_reduce < -998.0f) ? "" : std::to_string(val_seq_reduce)) + ","
            << ((val_par_reduce < -998.0f) ? "" : std::to_string(val_par_reduce)) + ","
            << ((val_gpu_reduce < -998.0f) ? "" : std::to_string(val_gpu_reduce)) + ","
            << diff_sg << "\n";
    }

    file.close();
    std::cout << "Vysledky byly ulozeny do: " << filename << std::endl;
}


ReadDexcomData DataReader::read() {
    auto d = ReadDexcomData();
    uint32_t counter = 0;
    std::string line;
    while (std::getline(*mFile, line)) {
        counter++;
        if ((counter < 2) || (counter < 14 && mDataType == DEXCOM))
            continue;
        // add DexcomRecord to d
        auto [index, timestamp, value] = parseLine(line);
        if (value < 0)
            continue;
        d.mData.emplace_back(index, timestamp, value);
        d.size++;
    };
    return std::move(d);
}

std::optional<int> DataConverter::minutes_since_midnight(const std::string_view& ts) {
    // Fast path: assume fixed positions "YYYY-MM-DDTHH:MM:SS" or "YYYY-MM-DD HH:MM:SS"
    //             0123456789012345678
    //             0         1
    // Need at least 16 chars to read HH:MM
    if (ts.size() >= 16) {
        // If 'T' or ' ' at index 10 and ':' at 13 and 16
        if ((ts[10] == 'T' || ts[10] == ' ') && ts[13] == ':') {
            int hh = 0, mm = 0;
            auto sv_hh = ts.substr(11, 2);
            auto sv_mm = ts.substr(14, 2);

            if (std::isdigit(sv_hh[0]) && std::isdigit(sv_hh[1]) &&
                std::isdigit(sv_mm[0]) && std::isdigit(sv_mm[1])) {

                // from_chars avoids locale issues
                auto to_int = [](std::string_view s)->std::optional<int>{
                    int out{};
                    auto res = std::from_chars(s.data(), s.data()+s.size(), out);
                    if (res.ec == std::errc{}) return out;
                    return std::nullopt;
                };

                auto oh = to_int(sv_hh);
                auto om = to_int(sv_mm);
                if (oh && om && *oh >= 0 && *oh < 24 && *om >= 0 && *om < 60) {
                    return *oh * 60 + *om;
                }
            }
        }
    }
    return std::nullopt;
}

std::optional<uint32_t> DataConverter::seconds_since_midnight(const std::string_view& ts) {
    // Fast path: assume fixed positions "YYYY-MM-DDTHH:MM:SS" or "YYYY-MM-DD HH:MM:SS"
    //             0123456789012345678
    //             0         1
    // Need at least 16 chars to read HH:MM
    if (ts.size() >= 16) {
        // If 'T' or ' ' at index 10 and ':' at 13 and 16
        if ((ts[10] == 'T' || ts[10] == ' ') && ts[13] == ':') {
            int hh = 0, mm = 0, ss = 0;
            auto sv_hh = ts.substr(11, 2);
            auto sv_mm = ts.substr(14, 2);
            auto sv_ss = ts.substr(17, 2);

            if (std::isdigit(sv_hh[0]) && std::isdigit(sv_hh[1]) &&
                std::isdigit(sv_mm[0]) && std::isdigit(sv_mm[1]) &&
                std::isdigit(sv_ss[0]) && std::isdigit(sv_ss[1])) {

                // from_chars avoids locale issues
                auto to_int = [](std::string_view s)->std::optional<int>{
                    int out{};
                    auto res = std::from_chars(s.data(), s.data()+s.size(), out);
                    if (res.ec == std::errc{}) return out;
                    return std::nullopt;
                };

                auto oh = to_int(sv_hh);
                auto om = to_int(sv_mm);
                auto os = to_int(sv_ss);
                if (oh && om && os &&
                    *oh >= 0 && *oh < 24 &&
                    *om >= 0 && *om < 60 &&
                    *os >= 0 && *os < 60) {
                    return *oh * 3600 + *om * 60 + *os;
                }
                }
        }
    }
    return std::nullopt;
}

std::optional<uint32_t> DataConverter::units_15ms_since_midnight(const std::string_view& ts) {
    // Fast path: assume fixed positions "YYYY-MM-DDTHH:MM:SS" or "YYYY-MM-DD HH:MM:SS"
    //             012345678901234567890
    //             0         1         2
    // Need at least 19 chars for HH:MM:SS
    if (ts.size() >= 23) {
        // If 'T' or ' ' at index 10 and ':' at 13 and 16
        if ((ts[10] == 'T' || ts[10] == ' ') && ts[13] == ':' && ts[16] == ':') {
            int hh = 0, mm = 0, ss = 0, ms = 0;

            auto sv_hh = ts.substr(11, 2);
            auto sv_mm = ts.substr(14, 2);
            auto sv_ss = ts.substr(17, 2);

            // Helper to parse integer safely
            auto to_int = [](std::string_view s) -> std::optional<int> {
                int out{};
                auto res = std::from_chars(s.data(), s.data() + s.size(), out);
                if (res.ec == std::errc{}) return out;
                return std::nullopt;
            };

            // Parse HH:MM:SS
            auto oh = to_int(sv_hh);
            auto om = to_int(sv_mm);
            auto os = to_int(sv_ss);

            if (oh && om && os &&
                *oh >= 0 && *oh < 24 &&
                *om >= 0 && *om < 60 &&
                *os >= 0 && *os < 60) {

                // Parse Milliseconds if present (e.g. .123 or ,123)
                // We look at index 19 for the separator
                if (ts.size() > 20 && (ts[19] == '.' || ts[19] == ',')) {
                    // Count actual digits to handle shorthand correctly
                    int digit_count = 0;
                    // We only care about up to 3 digits for milliseconds precision
                    for(size_t i = 20; i < ts.size() && i < 23; ++i) {
                        if(std::isdigit(ts[i])) digit_count++;
                        else break;
                    }

                    if (digit_count > 0) {
                        auto sv_ms = ts.substr(20, digit_count);
                        auto oms = to_int(sv_ms);
                        if (oms) {
                            ms = *oms;
                            // Normalize to milliseconds (ISO 8601 rule: .5 is 500ms, .05 is 50ms)
                            if (digit_count == 1) ms *= 100;
                            else if (digit_count == 2) ms *= 10;
                        }
                    }
                }

                // Calculate total milliseconds since midnight
                // (Hours * 3600 + Minutes * 60 + Seconds) * 1000 + Milliseconds
                uint32_t total_ms = (*oh * 3600 + *om * 60 + *os) * 1000 + ms;

                // Return the 15ms unit index
                return total_ms / 15;
            }
        }
    }
    return std::nullopt;
}

uint16_t DataConverter::slot_id_from_minutes(uint16_t minutes, uint16_t slot_minutes) {
    return minutes / slot_minutes; // 0..287 for 5-min slots
}

std::shared_ptr<DayValues> DataConverter::convertFromDexcomToDayValues(std::span<const ReadDexcomData> patients) {
    auto out = std::make_shared<DayValues>();
    std::vector<uint16_t> ids;
    ids.reserve(TIME_SLOTS_IN_DAY);

    for (const auto& data : patients) {
        // Map slot_id -> vector<float> values for THIS patient
        std::unordered_map<uint16_t, std::vector<float>> perSlot;
        perSlot.reserve(TIME_SLOTS_IN_DAY);

        for (const auto& rec : data.mData) {
            auto msm = minutes_since_midnight(rec.mTimestamp);
            if (!msm) {
                // Skip unparseable timestamps; could also log or collect stats.
                continue;
            }
            const int sid = slot_id_from_minutes(*msm, SLOT_LENGTH_IN_MINUTES);
            if (sid < 0 || sid > 10000) continue; // sanity
            perSlot[static_cast<uint16_t>(sid)].push_back(rec.mValue);
        }

        if (out->timeSlots.size() < 1) {
            // Build TimeSlot objects (sorted by slot id)
            for (const auto& [k, _] : perSlot) ids.push_back(k);
            std::ranges::sort(ids);

            out->timeSlots.reserve(ids.size());
            out->timeSlotMedianValues.reserve(ids.size());
            for (uint16_t sid : ids) {
                TimeSlot ts;
                ts.time_slot_id = sid;
                ts.patientMedianValues.clear();
                ts.patientData.clear();
                out->timeSlots.push_back(std::move(ts));
            }
        }


        for (uint16_t sid : ids) {
            // Single-patient dataset -> one PatientDataTimeSlot
            PatientDataTimeSlot p;
            p.time_slot_id = sid;
            p.values = std::move(perSlot[sid]); // move collected values

            out->timeSlots.at(sid).patientData.push_back(std::move(p));

            // Per-patient medians (here: one)
            out->timeSlots.at(sid).patientMedianValues.push_back(p.getMedianValue());

        }
    }

    return out;
}

DexcomData DataConverter::convertToFlatData(
    std::span<const ReadDexcomData> patients,
    uint32_t num_time_slots
) {
    DexcomData out;
    out.num_time_slots = num_time_slots;
    out.num_patients = static_cast<uint32_t>(patients.size());

    // Velikost = n_timeslots * Pacienti * n_days
    size_t total_size = (size_t)num_time_slots * out.num_patients * DexcomData::FIXED_NUM_DAYS;

    // Inicializace na -1.0f (padding).
    out.flat_data.assign(total_size, EMPTY_VALUE);

    // Alokace pro výsledky
    out.result_medians_seq.reserve(num_time_slots);
    out.result_medians_par.resize(num_time_slots);
    out.result_medians_par_nov_vect.resize(num_time_slots);
    out.result_medians_gpu.resize(num_time_slots);

    out.result_medians_seq_per_pat.reserve(num_time_slots * out.num_patients);
    out.result_medians_par_per_pat.resize(num_time_slots * out.num_patients);
    out.result_medians_par_nov_vect_per_pat.resize(num_time_slots * out.num_patients);
    out.result_medians_gpu_per_pat.resize(num_time_slots * out.num_patients);


    std::vector<uint8_t> day_counters(out.num_patients * num_time_slots, 0);

    for (size_t p_idx = 0; p_idx < patients.size(); ++p_idx) {
        for (const auto& rec : patients[p_idx].mData) {

             auto msm = minutes_since_midnight(rec.mTimestamp);
             if (!msm) continue;

            uint16_t sid = slot_id_from_minutes(*msm, SLOT_LENGTH_IN_MINUTES);
            if (sid >= num_time_slots) continue;

            size_t counter_idx = p_idx * num_time_slots + sid;
            uint8_t current_day_idx = day_counters[counter_idx];

            if (current_day_idx >= DexcomData::FIXED_NUM_DAYS) {
                continue;
            }
            // Vzorec: TS * (P * D) + P * D + d
            size_t flat_index = (size_t)sid * (out.num_patients * DexcomData::FIXED_NUM_DAYS) +
                                (size_t)p_idx * DexcomData::FIXED_NUM_DAYS +
                                (size_t)current_day_idx;

            out.flat_data[flat_index] = rec.mValue;

            day_counters[counter_idx]++;
        }
    }

    return out;
}

DexcomData DataConverter::convertToFlatDataInSeconds(
    std::span<const ReadDexcomData> patients,
    uint32_t num_time_slots,
    int32_t num_wanted_time_slots,
    uint32_t num_days
) {
    DexcomData out;
    out.num_time_slots = num_time_slots;
    out.num_days = num_days;
    out.num_patients = static_cast<uint32_t>(patients.size());

    // Velikost = n_timeslots * Pacienti * n_days
    size_t total_size = (size_t)num_time_slots * out.num_patients * num_days;

    out.flat_data.assign(total_size, EMPTY_VALUE);

    // Alokace pro výsledky
    out.result_medians_seq.reserve(num_time_slots);
    out.result_medians_par.resize(num_time_slots);
    out.result_medians_par_nov_vect.resize(num_time_slots);
    out.result_medians_gpu.resize(num_time_slots);

    out.result_medians_seq_per_pat.reserve(num_time_slots * out.num_patients);
    out.result_medians_par_per_pat.resize(num_time_slots * out.num_patients);
    out.result_medians_par_nov_vect_per_pat.resize(num_time_slots * out.num_patients);
    out.result_medians_gpu_per_pat.resize(num_time_slots * out.num_patients);

    if (num_wanted_time_slots > 0 || num_wanted_time_slots != num_time_slots) {
        out.updated_timeslots_seq.resize(num_wanted_time_slots);
        out.updated_timeslots_par.resize(num_wanted_time_slots);
        out.updated_timeslots_par_non_vect.resize(num_wanted_time_slots);
        out.updated_timeslots_gpu.resize(num_wanted_time_slots);
    }

    // std::cout << "Size of " << out.result_medians_par_per_pat.size() << std::endl;

    std::vector<uint8_t> day_counters(out.num_patients * num_time_slots, 0);

    uint32_t max = 0;
    for (size_t p_idx = 0; p_idx < patients.size(); p_idx++) {
        for (const auto& rec : patients[p_idx].mData) {

             auto msm = seconds_since_midnight(rec.mTimestamp);
            // std::cout << "Checking seconds_since_midnight " << std::endl;

             if (!msm) continue;


            uint32_t sid = *msm;
            // std::cout << "Checking seconds " << sid << std::endl;
            if (sid >= num_time_slots) continue;

            size_t counter_idx = p_idx * num_time_slots + sid;
            uint8_t current_day_idx = day_counters[counter_idx];

            // std::cout << "Checking days " << current_day_idx << std::endl;

            if (current_day_idx >= num_days) {
                continue;
            }

            // Vzorec: TS * (P * D) + P * D + d
            size_t flat_index = (size_t)sid * (out.num_patients * num_days) +
                                (size_t)p_idx * num_days +
                                (size_t)current_day_idx;
            if (flat_index > max)
                max = flat_index;

            // if (sid > 80000) {
            //     std::cout << "Putting " << rec.mValue << " on index " << flat_index << std::endl;
            //     std::cout << "Second: " << sid << ", patientd: " << p_idx << ", current_day_idx: " << static_cast<size_t>(current_day_idx) << std::endl;
            //
            // }
            if (out.flat_data[flat_index] > 0) {
                std::cerr << "On index " << flat_index << " is already a value: " << out.flat_data[flat_index] << ". Patient: " << p_idx << ", s: " << sid << std::endl;
                break;
            }

            out.flat_data[flat_index] = rec.mValue;

            day_counters[counter_idx]++;
        }
    }
    // std::cout << "Max flat index: " << max << std::endl;

    return out;
}

DexcomData DataConverter::convertToFlatDataInMilliseconds(
    std::span<const ReadDexcomData> patients,
    uint32_t num_time_slots,
    int32_t num_wanted_time_slots,
    uint32_t num_days
) {
    DexcomData out;
    out.num_time_slots = num_time_slots;
    out.num_days = num_days;
    out.num_patients = static_cast<uint32_t>(patients.size());


    // pro 4 pacientz - 184 320 000
    size_t total_size = (size_t)num_time_slots * out.num_patients * num_days;


    try {
        out.flat_data.assign(total_size, EMPTY_VALUE);
    } catch (const std::bad_alloc& e) {
        std::cerr << "Chyba alokace paměti pro flat_data: " << e.what() << std::endl;
        std::cerr << "Požadováno prvků: " << total_size << " (" << (total_size * 4 / 1024 / 1024) << " MB)" << std::endl;
        throw;
    }

    out.result_medians_seq.reserve(num_time_slots);
    out.result_medians_par.resize(num_time_slots);
    out.result_medians_par_nov_vect.resize(num_time_slots);
    out.result_medians_gpu.resize(num_time_slots);

    size_t per_patient_size = (size_t)num_time_slots * out.num_patients;
    out.result_medians_seq_per_pat.reserve(per_patient_size);
    out.result_medians_par_per_pat.resize(per_patient_size);
    out.result_medians_par_nov_vect_per_pat.resize(per_patient_size);
    out.result_medians_gpu_per_pat.resize(per_patient_size);

    if (num_wanted_time_slots > 0 || num_wanted_time_slots != num_time_slots) {
        out.updated_timeslots_seq.resize(num_wanted_time_slots);
        out.updated_timeslots_par.resize(num_wanted_time_slots);
        out.updated_timeslots_par_non_vect.resize(num_wanted_time_slots);
        out.updated_timeslots_gpu.resize(num_wanted_time_slots);
    }

    std::vector<uint8_t> day_counters;
    try {
        day_counters.resize((size_t)out.num_patients * num_time_slots, 0);
    } catch (const std::bad_alloc& e) {
        std::cerr << "Chyba alokace paměti pro day_counters." << std::endl;
        throw;
    }

    uint32_t max_idx = 0;

    for (size_t p_idx = 0; p_idx < patients.size(); p_idx++) {
        for (const auto& rec : patients[p_idx].mData) {

            auto msm = units_15ms_since_midnight(rec.mTimestamp);

            if (!msm) continue;

            uint32_t sid = *msm;

            if (sid >= num_time_slots) continue;

            size_t counter_idx = (size_t)p_idx * num_time_slots + sid;
            uint8_t current_day_idx = day_counters[counter_idx];

            if (current_day_idx >= num_days) {
                continue;
            }

            size_t flat_index = (size_t)sid * (out.num_patients * num_days) +
                                (size_t)p_idx * num_days +
                                (size_t)current_day_idx;

            if (flat_index > max_idx) max_idx = flat_index;

            if (out.flat_data[flat_index] > 0) {
                // Toto by se nemělo stát díky logice day_counters, ledaže by day_counters selhalo
                // nebo data obsahovala duplicity pro stejný čas.
            } else {
                out.flat_data[flat_index] = rec.mValue;
                day_counters[counter_idx]++;
            }
        }
    }

    std::cout << "Max flat index usage: " << max_idx << " / " << total_size << std::endl;

    return out;
}
