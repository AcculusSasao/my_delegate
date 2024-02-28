#include <string>
#include <vector>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/external/external_delegate_interface.h"
#include "tensorflow/lite/delegates/utils/my_delegate/my_delegate.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace tools {

TfLiteDelegate* CreateMyDelegateFromOptions(
    const char* const* options_keys,
    const char* const* options_values,
    size_t num_options)
{
    MyDelegateOptions options = TfLiteMyDelegateOptionsDefault();

    std::vector<const char*> argv;
    argv.reserve(num_options + 1);
    constexpr char kMyDelegateParsing[] = "my_delegate_parsing";
    argv.push_back(kMyDelegateParsing);

    std::vector<std::string> option_args;
    option_args.reserve(num_options);
    for (int i = 0; i < num_options; ++i) {
        option_args.emplace_back("--");
        option_args.rbegin()->append(options_keys[i]);
        option_args.rbegin()->push_back('=');
        option_args.rbegin()->append(options_values[i]);
        argv.push_back(option_args.rbegin()->c_str());
    }

    std::vector<tflite::Flag> flag_list = {
        tflite::Flag::CreateFlag("param_a", &options.param_a, "parameter A"),
        tflite::Flag::CreateFlag("param_b", &options.param_b, "parameter B"),
    };

    int argc = num_options + 1;
    if (!tflite::Flags::Parse(&argc, argv.data(), flag_list)) {
        return nullptr;
    }

    TFLITE_LOG(INFO) << "param_a: " << options.param_a;
    TFLITE_LOG(INFO) << "param_b: " << options.param_b;
    return TfLiteMyDelegateCreate(&options);
}

}  // namespace tools
}  // namespace tflite

extern "C" {

extern TFL_EXTERNAL_DELEGATE_EXPORT TfLiteDelegate* tflite_plugin_create_delegate(
    const char* const* options_keys,
    const char* const* options_values,
    size_t num_options,
    void (*report_error)(const char*))
{
    return tflite::tools::CreateMyDelegateFromOptions(options_keys, options_values, num_options);
}

TFL_EXTERNAL_DELEGATE_EXPORT void tflite_plugin_destroy_delegate(
    TfLiteDelegate* delegate)
{
    TfLiteMyDelegateDelete(delegate);
}

}  // extern "C"
