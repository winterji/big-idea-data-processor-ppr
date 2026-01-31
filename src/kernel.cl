// kernel.cl

#define EMPTY_VALUE -1.0e30f
#define VALID_THRESHOLD (EMPTY_VALUE + 100.0f)

// Makro pro Compare-And-Swap
#define CAS(a, b) { \
    float tmp_a = a; \
    float tmp_b = b; \
    a = (tmp_a < tmp_b) ? tmp_a : tmp_b; \
    b = (tmp_a < tmp_b) ? tmp_b : tmp_a; \
}

void shell_sort_global(__global float* data, int start_idx, int count) {
    // Gap sequence: n/2, n/4, ... 1
    for (int gap = count / 2; gap > 0; gap /= 2) {
        // Gapped insertion sort
        for (int i = gap; i < count; i += 1) {
            float temp = data[start_idx + i];
            int j;
            // Posouváme prvky, dokud nenajdeme místo pro temp
            for (j = i; j >= gap && data[start_idx + j - gap] > temp; j -= gap) {
                data[start_idx + j] = data[start_idx + j - gap];
            }
            data[start_idx + j] = temp;
        }
    }
}

void sort(float* arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                float temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

__kernel void median_kernel(
    __global const float* data,
    __global float* result_medians,
    const int num_days,
    const int total_patients_slots
) {
    int gid = get_global_id(0);

    if (gid >= total_patients_slots) {
        return;
    }

    long offset = (long)gid * (long)num_days;

    float v0 = data[offset + 0];
    float v1 = data[offset + 1];
    float v2 = data[offset + 2];
    float v3 = data[offset + 3];
    float v4 = data[offset + 4];
    float v5 = data[offset + 5];
    float v6 = data[offset + 6];
    float v7 = data[offset + 7];

    CAS(v0, v1);
    CAS(v2, v3);
    CAS(v4, v5);
    CAS(v6, v7);

    CAS(v0, v2);
    CAS(v1, v3);
    CAS(v4, v6);
    CAS(v5, v7);

    CAS(v1, v2);
    CAS(v5, v6);
    CAS(v0, v4);
    CAS(v3, v7);

    CAS(v1, v5); CAS(v2, v6);
    CAS(v1, v4); CAS(v3, v6);
    CAS(v2, v4); CAS(v3, v5);
    CAS(v3, v4); CAS(v1, v2); CAS(v5, v6);

    int valid_count = 0;

    valid_count += (v0 > VALID_THRESHOLD);
    valid_count += (v1 > VALID_THRESHOLD);
    valid_count += (v2 > VALID_THRESHOLD);
    valid_count += (v3 > VALID_THRESHOLD);
    valid_count += (v4 > VALID_THRESHOLD);
    valid_count += (v5 > VALID_THRESHOLD);
    valid_count += (v6 > VALID_THRESHOLD);
    valid_count += (v7 > VALID_THRESHOLD);

    float median = EMPTY_VALUE;

    if (valid_count > 0) {
        int start_idx = 8 - valid_count;
        int mid = valid_count / 2;
        int target = start_idx + mid;

        float sorted[8];
        sorted[0] = v0; sorted[1] = v1; sorted[2] = v2; sorted[3] = v3;
        sorted[4] = v4; sorted[5] = v5; sorted[6] = v6; sorted[7] = v7;

        median = sorted[target];

        if (valid_count % 2 == 0) {
            float prev = sorted[target - 1];
            median = (prev + median) / 2.0f;
        }
    }

    result_medians[gid] = median;
}

__kernel void reduce_patients_kernel(
    __global const float* input_matrix,
    __global float* output_medians,
    const int num_patients
) {
    int ts = get_global_id(0); // time_slot

    #define MAX_BUFFER_SIZE 16
    float buffer[MAX_BUFFER_SIZE];

    int valid_count = 0;
    long offset = (long)ts * (long)num_patients;

    for (int p = 0; p < num_patients; ++p) {
        float val = input_matrix[offset + p];
        if (val > VALID_THRESHOLD && valid_count < MAX_BUFFER_SIZE) {
            buffer[valid_count++] = val;
        }
    }

    float result = EMPTY_VALUE;
    if (valid_count > 0) {
        sort(buffer, valid_count);

        int mid = valid_count / 2;
        result = buffer[mid];

        // sudy pocet
        if (valid_count % 2 == 0) {
            result = (buffer[mid - 1] + result) / 2.0f;
        }
    }

    output_medians[ts] = result;
}

__kernel void reduce_slots_kernel(
    __global float* input_slots,
    __global float* output_final,
    const int num_total_slots,
    const int num_wanted_slots
) {
    int i = get_global_id(0); // wanted_time_slot
    if (i >= num_wanted_slots) return;

    long start_idx_l = ((long)i * num_total_slots) / num_wanted_slots;
    long end_idx_l = ((long)(i + 1) * num_total_slots) / num_wanted_slots;

    int start_idx = (int)start_idx_l;
    int end_idx = (int)end_idx_l;
    if (end_idx > num_total_slots) end_idx = num_total_slots;

    //komprese dat
    int valid_count = 0;
    int write_ptr = start_idx;

    for (int read_ptr = start_idx; read_ptr < end_idx; ++read_ptr) {
        float val = input_slots[read_ptr];

        if (val > VALID_THRESHOLD) {
            if (read_ptr != write_ptr) {
                input_slots[write_ptr] = val;
            }
            write_ptr++;
            valid_count++;
        }
    }

    float result = EMPTY_VALUE;

    if (valid_count > 0) {
        shell_sort_global(input_slots, start_idx, valid_count);

        int mid = valid_count / 2;
        result = input_slots[start_idx + mid];

        if (valid_count % 2 == 0) {
            float prev = input_slots[start_idx + mid - 1];
            result = (prev + result) / 2.0f;
        }
    }

    // #define MAX_SLOT_BUFFER 512
    // float buffer[MAX_SLOT_BUFFER];
    // int valid_count = 0;

    // for (int k = start_idx; k < end_idx; ++k) {
    //     float val = input_slots[k];
    //     if (val > 0.001f && valid_count < MAX_SLOT_BUFFER) {
    //         buffer[valid_count++] = val;
    //     }
    // }

    // float result = 0.0f;
    // if (valid_count > 0) {
    //     sort(buffer, valid_count);

    //     int mid = valid_count / 2;
    //     result = buffer[mid];

    //     if (valid_count % 2 = = 0) {
    //         result = (buffer[mid - 1] + result) / 2.0f;
    //     }
    // }

    output_final[i] = result;
}