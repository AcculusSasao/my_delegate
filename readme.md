# Tensorflow Lite Delegate sample code

解説ブログ: https://link.medium.com/8nApnBGjyHb
  
TFLiteのデリゲートを自作するサンプルコードです。  
参照:  https://www.tensorflow.org/lite/performance/implementing_delegate  

## 準備

tensorflow をcloneし、  
その tensorflow/lite/delegates/utils 下に本リポジトリをcloneします  

```
git clone -b r2.15 https://github.com/tensorflow/tensorflow.git

cd tensorflow/tensorflow/lite/delegates/utils
git clone https://github.com/AcculusSasao/my_delegate
```

必要な物(libtensorflowlite.so, benchmark_modelツール)、モデル、テスト画像のダウンロード
```
scripts/prepare.sh
```

## デリゲートのビルド

```
scripts/build.sh
```

## デリゲートを使った実行

ツールbenchmark_modelを使って、ランダム入力で実行

```
scripts/run_benchmark.sh mobilenet_v1_0.25_128.tflite 
```

pythonでパンダ画像を入力してmobilenetv1を実行

```
scripts/infer_delegate.sh
```
