source remove_cache.sh

source download.sh

python2 converter.py

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

python3 demo.py
