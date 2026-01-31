//
// Created by Jiří Winter on 02.11.2025.
//

#ifndef PPR_SEMESTRALKA_DATAREADER_H
#define PPR_SEMESTRALKA_DATAREADER_H

#include <memory>
#include <string>
#include <span>
#include "ReadDexcomData.h"


class DataReader {
private:
    std::string mFilename;
    std::unique_ptr<std::ifstream> mFile;
    const DataType mDataType;
public:
    DataReader(const std::string& filename, DataType dataType);
    ~DataReader();

    std::tuple<uint32_t, std::string, float> parseLine(const std::string& line);
    ReadDexcomData read();
    bool isOpen() const { return mFile->is_open(); }
};

class DataConverter {
public:
    // static std::shared_ptr<DayValues> convertFromDexcomToDayValues(const DexcomData& data);
    static std::shared_ptr<DayValues> convertFromDexcomToDayValues(std::span<const ReadDexcomData> patients);
    // variadic wrapper (no copies)
    template <class... Ds>
    static std::shared_ptr<DayValues> convertFromDexcomToDayValues(const ReadDexcomData& first, const Ds&... rest) {
        const ReadDexcomData* arr[] = { &first, (&rest)... };
        // Build a temporary vector of values to create a span
        // (we dereference immediately inside the impl)
        std::vector<ReadDexcomData> tmp;
        tmp.reserve(sizeof...(rest) + 1);
        for (auto* p : arr)
            tmp.push_back(*p);              // copies; if you want no copies, see note below
        return convertFromDexcomToDayValues(std::span<const ReadDexcomData>(tmp));
    }

    static std::optional<uint32_t> units_15ms_since_midnight(const std::string_view& ts);

    static DexcomData convertToFlatDataInMilliseconds(
        std::span<const ReadDexcomData> patients,
        uint32_t num_time_slots,
        int32_t num_wanted_time_slots,
        uint32_t num_days
    );

    /** If number of wanted time slots is the same as provided in data, can be left blank
     *
     * @param patients
     * @param num_time_slots
     * @param num_wanted_time_slots
     * @param num_days
     * @return
     */
    static DexcomData convertToFlatDataInSeconds(
        std::span<const ReadDexcomData> patients,
        uint32_t num_time_slots,
        int32_t num_wanted_time_slots = -1,
        uint32_t num_days = DexcomData::FIXED_NUM_DAYS
    );
    static DexcomData convertToFlatData(std::span<const ReadDexcomData> patients, uint32_t num_time_slots = DexcomData::NUM_TIME_SLOTS);
    template <class... Ds>
    static DexcomData convertToFlatData(const ReadDexcomData& first, const Ds&... rest) {
        const ReadDexcomData* arr[] = { &first, (&rest)... };
        std::vector<ReadDexcomData> tmp;
        tmp.reserve(sizeof...(rest) + 1);
        for (auto* p : arr)
            tmp.push_back(*p);
        return convertToFlatData(std::span<const ReadDexcomData>(tmp));
    };
    static std::optional<int> minutes_since_midnight(const std::string_view& ts);
    static std::optional<uint32_t> seconds_since_midnight(const std::string_view& ts);
    static uint16_t slot_id_from_minutes(uint16_t minutes, uint16_t slot_minutes);
};

#endif //PPR_SEMESTRALKA_DATAREADER_H