#include "llama.cpp/tools/mtmd/mtmd.h"
#include "llama.cpp/tools/mtmd/mtmd-helper.h"

struct mtmd_rs_memory_usage {
    const char * device_name;
    size_t bytes;
    bool host;
};

#ifdef __cplusplus
extern "C" {
#endif

size_t mtmd_rs_get_memory_usage(
    const char * mmproj_fname,
    struct mtmd_context_params ctx_params,
    struct mtmd_rs_memory_usage * usage,
    size_t capacity);

#ifdef __cplusplus
}
#endif
