#pragma once
#include "common.h"
#include <string>
#include <vector>
#include <regex>

// common debug functions and structs

// Intended to use as callback for ggml_backend_sched_eval_callback
// prints tensors that are processed in the computation graph
// by default prints all tensors, but can be configured by creating a `common_debug_cb_user_data` instance with
// non-empty filter_patterns. See examples/debug.ccp for possible usage patterns
// `common_debug_cb_user_data` contains `abort_on_nan` flag that determines whether an error should be thrown whenever a NaN is encountered
// in a tensor (useful for stopping debug sessions on first erroneous tensor)
// The callback data will be passed as the third parameter (user_data)
bool common_debug_cb_eval(struct ggml_tensor * t, bool ask, void * user_data);

struct common_debug_cb_user_data {
    std::vector<uint8_t>    data;
    std::vector<std::regex> tensor_filters;
    bool                    abort_on_nan{false};

    common_debug_cb_user_data() = default;

    common_debug_cb_user_data(common_params & params, const std::vector<std::string> & filter_patterns, bool abort_on_nan = false) {
        for (const auto & pattern : filter_patterns) {
            try {
                std::string anchored_pattern = "^" + pattern;
                tensor_filters.emplace_back(anchored_pattern, std::regex::optimize);
            } catch (const std::regex_error & e) {
                throw std::runtime_error("Invalid regex pattern '" + pattern + "': " + e.what());
            }
        }
        this->abort_on_nan = abort_on_nan;

        params.cb_eval           = common_debug_cb_eval;
        params.cb_eval_user_data = this;
    }
};
