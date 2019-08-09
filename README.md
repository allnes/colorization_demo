# Colorization Demo
1) Download networks colorization - repository [this](https://github.com/richzhang/colorization)
```
source download.sh
```
2) Update networks for conversion (required Caffe)
```
python2 converter.py
```
3) Convert `Caffe -> IR` with help Intel Model Optimizer
```
mo.py \
  --framework=caffe \
  --data_type=FP32 \
  --input_shape=[1,1,224,224] \
  --input=data_l \
  --mean_values=data[50.0,50.0,50.0] \
  --scale_values=data[255] \
  --output=class8_ab \
  --input_model colorization.caffemodel \
  --input_proto colorization.prototxt
```
