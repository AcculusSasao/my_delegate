#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/tools/logging.h"

#include "tensorflow/lite/delegates/utils/my_delegate/my_delegate.h"

namespace tflite {
namespace my_test {

static void printTensor(const TfLiteTensor& t, std::string prefix=""){
    std::string str = "";
    for(int i=0; i<t.dims->size; i++) str += std::to_string(t.dims->data[i]) + ",";
    TFLITE_LOG(INFO) << prefix << t.name << " type=" << t.type << ", bytes=" << t.bytes << ", dims=[" << str << "]";
}
static void printConvParams(const TfLiteConvParams* p, std::string prefix=""){
    char str[128];
    snprintf(str,128,"ConvParams pad=%d, stride_w=%d,h=%d, act=%d, dilation_w=%d,h=%d, bias_type=%d",
        p->padding, p->stride_width, p->stride_height, p->activation,
        p->dilation_width_factor, p->dilation_height_factor, p->quantized_bias_type);
    TFLITE_LOG(INFO) << prefix << str;
}

class MyDelegateKernel : public SimpleDelegateKernelInterface {
public:
    explicit MyDelegateKernel(const MyDelegateOptions& options) : options_(options) {}

    TfLiteStatus Init(TfLiteContext* context, const TfLiteDelegateParams* params) override {
        //TFLITE_LOG(INFO) << "DelegateKernel Init " << this;
        inputs_.resize(params->nodes_to_replace->size);
        outputs_.resize(params->nodes_to_replace->size);
        builtin_code_.resize(params->nodes_to_replace->size);
        builtin_data_.resize(params->nodes_to_replace->size);
        for(int i=0; i<params->nodes_to_replace->size; i++){
            const int node_index = params->nodes_to_replace->data[i];
            TfLiteNode* delegated_node = nullptr;
            TfLiteRegistration* delegated_node_registration = nullptr;
            TF_LITE_ENSURE_EQ(
                context,
                context->GetNodeAndRegistration(context, node_index, &delegated_node, &delegated_node_registration),
                kTfLiteOk);
            inputs_[i].push_back(delegated_node->inputs->data[0]);
            inputs_[i].push_back(delegated_node->inputs->data[1]);
            inputs_[i].push_back(delegated_node->inputs->data[2]);
            outputs_[i].push_back(delegated_node->outputs->data[0]);
            builtin_code_[i] = delegated_node_registration->builtin_code;
            builtin_data_[i] = delegated_node->builtin_data;
            // print params
            if(delegated_node_registration->builtin_code == kTfLiteBuiltinConv2d){
                auto* conv_params = reinterpret_cast<TfLiteConvParams*>(delegated_node->builtin_data);
                printConvParams(conv_params);
            }
        }
        return kTfLiteOk;
    }

    TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
        //TFLITE_LOG(INFO) << "DelegateKernel Prepare " << this;
        return kTfLiteOk;
    }

    TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
        //TFLITE_LOG(INFO) << "DelegateKernel Eval " << this;
        for(int i=0; i<inputs_.size(); i++){
            if(builtin_code_[i] == kTfLiteBuiltinConv2d){
                auto& input0 = context->tensors[inputs_[i][0]];
                auto& input1 = context->tensors[inputs_[i][1]];
                auto& input2 = context->tensors[inputs_[i][2]];
                auto& output = context->tensors[outputs_[i][0]];
                auto* conv_params = reinterpret_cast<TfLiteConvParams*>(builtin_data_[i]);
                TF_LITE_ENSURE_EQ(
                    context,
                    Conv2d(context, builtin_code_[i], &input0, &input1, &input2, &output, conv_params),
                    kTfLiteOk);
            }
        }
        return kTfLiteOk;
    }

private:
    TfLiteStatus Conv2d(TfLiteContext* context, int builtin_code,
        const TfLiteTensor* input, const TfLiteTensor* filter, const TfLiteTensor* bias,
        TfLiteTensor* output, const TfLiteConvParams* par)
    {
        //TFLITE_LOG(INFO) << "start Conv2d";
        auto* d_input = GetTensorData<float>(input);
        auto* d_filter = GetTensorData<float>(filter);
        auto* d_bias = GetTensorData<float>(bias);
        auto* d_output = GetTensorData<float>(output);
        
        // input: [batch, in_height, in_width, in_ch]
        // filter: [out_ch, filter_height, filter_width, in_ch]
        // bias: [out_ch]
        // output: [batch, out_height, out_width, out_ch]
        const int batch = input->dims->data[0];
        const int in_height = input->dims->data[1];
        const int in_width = input->dims->data[2];
        const int in_ch = input->dims->data[3];
        const int out_ch = filter->dims->data[0];
        const int filter_height = filter->dims->data[1];
        const int filter_width = filter->dims->data[2];
        const int out_height = output->dims->data[1];
        const int out_width = output->dims->data[2];
        
        // batch = 1 only.
        // padding = 1 'same' only.
        // activation = 3 'kTfLiteActRelu6' or 0 'kTfLiteActNone'.
        if(batch != 1 || par->padding != kTfLitePaddingSame || 
            (par->activation != kTfLiteActRelu6 && par->activation != kTfLiteActNone)){
            return kTfLiteError;
        }
        
        for(int y=0; y<in_height; y+=par->stride_height){
            for(int x=0; x<in_width; x+=par->stride_width){
                auto* _bias = d_bias;
                for(int oc=0; oc<out_ch; oc++){
                    float out = *_bias++;
                    for(int ic=0; ic<in_ch; ic++){
                        for(int fy=0; fy<filter_height; fy++){
                            const int iy = y - filter_height/2 + fy;
                            for(int fx=0; fx<filter_width; fx++){
                                const int ix = x - filter_width/2 + fx;
                                // input: [1, iy, ix, ic]
                                // filter: [oc, fy, fx, ic]
                                if(iy >= 0 && ix >= 0 && iy < in_height && ix < in_width)
                                    out += d_input[iy * in_width * in_ch + ix * in_ch + ic]
                                    * d_filter[oc * filter_height * filter_width * in_ch + fy * filter_width * in_ch + fx * in_ch + ic];
                            }
                        }
                    }
                    if(par->activation == kTfLiteActRelu6){
                        out = std::min<float>(std::max<float>(0, out), 6);
                    }
                    *d_output++ = out;
                }
            }
        }

        return kTfLiteOk;
    }

    const MyDelegateOptions options_;
    std::vector<std::vector<int>> inputs_, outputs_;
    std::vector<int> builtin_code_;
    std::vector<void*> builtin_data_;
};

class MyDelegate : public SimpleDelegateInterface {
public:
    explicit MyDelegate(const MyDelegateOptions& options)
        : options_(options) {}
    bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                    const TfLiteNode* node,
                                    TfLiteContext* context) const override {
        // print node info
        TFLITE_LOG(INFO) << "builtin_code=" << registration->builtin_code << ", name=" << registration->custom_name;
        for(int i=0; i<node->inputs->size; i++){
            char str[10];
            snprintf(str,10,"  in[%d] ",i);
            auto& tensor = context->tensors[node->inputs->data[i]];
            printTensor(tensor, str);
        }
        for(int i=0; i<node->outputs->size; i++){
            char str[10];
            snprintf(str,10,"  out[%d] ",i);
            auto& tensor = context->tensors[node->outputs->data[i]];
            printTensor(tensor, str);
        }

        // Supports Conv2d
        if(registration->builtin_code != kTfLiteBuiltinConv2d){
            return false;
        }
        // Supports float32
        for(int i=0; i<node->inputs->size; i++){
            auto& tensor = context->tensors[node->inputs->data[i]];
            if(tensor.type != kTfLiteFloat32) return false;
        }
        TFLITE_LOG(INFO) << "  -> supported.";
        return true;
    }

    TfLiteStatus Initialize(TfLiteContext* context) override { return kTfLiteOk; }

    const char* Name() const override {
        static constexpr char kName[] = "MyDelegate";
        return kName;
    }

    std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
        override {
        return std::make_unique<MyDelegateKernel>(options_);
    }

    SimpleDelegateInterface::Options DelegateOptions() const override {
        return SimpleDelegateInterface::Options();
    }

private:
    const MyDelegateOptions options_;
};

}  // namespace my_test
}  // namespace tflite

MyDelegateOptions TfLiteMyDelegateOptionsDefault() {
    MyDelegateOptions options = {0};
    options.param_a = 0;
    options.param_b = 0;
    return options;
}

TfLiteDelegate* TfLiteMyDelegateCreate(const MyDelegateOptions* options) {
    std::unique_ptr<tflite::my_test::MyDelegate> my(
        new tflite::my_test::MyDelegate(
            options ? *options : TfLiteMyDelegateOptionsDefault()));
    return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(my));
}

void TfLiteMyDelegateDelete(TfLiteDelegate* delegate) {
    tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
