#include "wrapper_mtmd.h"

#include "ggml-backend.h"

#include <cstring>
#include <map>

extern "C" size_t mtmd_rs_get_memory_usage(
    const char * mmproj_fname,
    struct mtmd_context_params ctx_params,
    struct mtmd_rs_memory_usage * usage,
    size_t capacity) {
    if (mmproj_fname == nullptr) {
        return 0;
    }

    const std::map<ggml_backend_dev_t, size_t> measured =
        mtmd_get_memory_usage(mmproj_fname, ctx_params);
    if (usage == nullptr || capacity == 0) {
        return measured.size();
    }

    size_t index = 0;
    for (const auto & [device, bytes] : measured) {
        if (index == capacity) {
            break;
        }
        usage[index++] = {
            ggml_backend_dev_name(device),
            bytes,
            ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU,
        };
    }
    return measured.size();
}
