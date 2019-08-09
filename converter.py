import numpy as np
import caffe
import google.protobuf.text_format as text_format

proto_name  = 'colorization.prototxt'
model_name  = 'colorization.caffemodel'

net = caffe.Net(proto_name, model_name, caffe.TEST)
points = np.load('points.npy')
net.params['class8_ab'][0].data[:,:,0,0] = points.transpose((1,0))
net.forward()
net.save(model_name)

net = caffe.proto.caffe_pb2.NetParameter()
with open(proto_name, 'r') as proto_file:
    text_format.Merge(proto_file.read(), net)

new_net = caffe.proto.caffe_pb2.NetParameter()
for layer in net.layer:
    if layer.name != r'Silence':
        curr_layer = new_net.layer.add()
        curr_layer.CopyFrom(layer)

with open(proto_name, 'w') as proto_file:
    proto_file.write(text_format.MessageToString(new_net))
