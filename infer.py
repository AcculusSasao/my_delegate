import argparse
import cv2
import numpy as np
import sys
from imagenet import create_readable_names_for_imagenet_labels
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite.Interpreter

def normalize_image(bgr_img: np.array, target_width: int, target_height: int, dtype: int,
                    mean: float = 128, std : float = 128):
    if target_width != target_height:
        raise ValueError('target_width must be same with target_height.')
    h, w = bgr_img.shape[:2]
    if h <= w:
        srcimg = np.zeros((w, w, 3), dtype=np.uint8)
        e = (w - h) // 2
        srcimg[e:e+h] = bgr_img
    else:
        srcimg = np.zeros((h, h, 3), dtype=np.uint8)
        e = (h - w) // 2
        srcimg[:, e:e+w] = bgr_img
    img = cv2.resize(srcimg, dsize=(target_width, target_height))
    data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = data.astype(dtype)
    data = ((data - mean) / std)
    return data[np.newaxis, :]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='panda.jpg', help='input image file')
    parser.add_argument('-m', '--model', default='mobilenet_v1_0.25_128.tflite', help='tflite model file')
    parser.add_argument('-d', '--delegate', default=None, help='external delegate file')
    parser.add_argument('-o', '--delegate_options', default='param_a:2;param_b:3', help='external delegate options string with ;and:')
    parser.add_argument('-n', '--num_threads', type=int, default=4, help='num_threads')
    args = parser.parse_args()
    
    delegate = None
    delegate_options = {}
    if args.delegate_options is not None:
        options = args.delegate_options.split(';')
        for opt in options:
            words = opt.split(':')
            if len(words) == 2:
                delegate_options[words[0]] = words[1].strip()
            else:
                print('invalid option', options)
                sys.exit(-1)
    if args.delegate is not None:
        delegate = [tflite.load_delegate(args.delegate, delegate_options)]
        print('use delegate:', args.delegate, 'with options:', delegate_options)

    interpreter = tflite.Interpreter(model_path=args.model, experimental_delegates=delegate, num_threads=args.num_threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape'][2]
	
    bgr_img = cv2.imread(args.input)
    if bgr_img is None:
        print('fail to open', args.input)
        sys.exit(-1)
    input_data = normalize_image(bgr_img, input_shape, input_shape, np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index']).flatten()

    labels = create_readable_names_for_imagenet_labels()
    if(len(labels) != len(out)):
        print('Different size of labels {} != out {}'.format(len(labels), len(out)))
    
    print('result top5:')
    top = sorted([[idx, value] for idx, value in enumerate(out)], key=lambda x: x[1], reverse=True)
    for t in top[:5]:
        print(t[1], ':', labels[t[0]], ' (', t[0], ')')
