#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_MY_DELEGATE_MY_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_MY_DELEGATE_MY_DELEGATE_H_
#include <memory>
#include "tensorflow/lite/core/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int param_a;
    int param_b;
} MyDelegateOptions;

MyDelegateOptions TfLiteMyDelegateOptionsDefault();
TfLiteDelegate* TfLiteMyDelegateCreate(const MyDelegateOptions* options);
void TfLiteMyDelegateDelete(TfLiteDelegate* delegate);

#ifdef __cplusplus
}
#endif

#endif
