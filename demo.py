from openvino.inference_engine import IENetwork, IECore
import cv2 as cv
import numpy as np

scale = 255
mean  = 50

load_net = IENetwork(model="colorization.xml", weights="colorization.bin")
exec_net = IECore().load_network(network=load_net, device_name="CPU")

input_blob  = next(iter(load_net.inputs))
output_blob = next(iter(load_net.outputs))

n, c, h_in, w_in = load_net.inputs[input_blob].shape
images = np.ndarray(shape=(n, c, h_in, w_in))

if 1:
    cap = cv.VideoCapture('test.mp4')
else:
    cap = cv.VideoCapture(0)

while cv.waitKey(1) < 0:
    has_frame, original_frame = cap.read()
    (h_orig, w_orig) = original_frame.shape[:2]
    if not has_frame:
        cv.waitKey()
        break

    # preprocessing frame
    frame = cv.cvtColor(cv.cvtColor(original_frame, cv.COLOR_RGB2GRAY), cv.COLOR_GRAY2RGB)
    img_rgb = (frame[:,:,[2, 1, 0]] * 1.0 / scale).astype(np.float32)
    img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
    img_l = img_lab[:,:,0]

    img_rs = cv.resize(img_rgb, (w_in, h_in))
    img_lab_rs = cv.cvtColor(img_rs, cv.COLOR_RGB2Lab)
    img_l_rs = img_lab_rs[:,:,0]
    img_l_rs -= mean

    for i in range(n):
        images[i] = img_l_rs

    # network inference
    res = exec_net.infer(inputs={input_blob: images})

    # get result
    out = res[output_blob][0,:,:,:].transpose((1,2,0))
    out = cv.resize(out, (w_orig, h_orig))
    img_lab_out = np.concatenate((img_l[:,:,np.newaxis],out),axis=2)
    img_bgr_out = np.clip(cv.cvtColor(img_lab_out, cv.COLOR_Lab2BGR), 0, 1)

    # show result
    show_size = (640, 480)
    frame = cv.resize(frame, show_size)
    original_frame = cv.resize(original_frame, show_size)
    colorized_frame = cv.resize(img_bgr_out, show_size)
    cv.imshow('origin', original_frame)
    cv.imshow('gray', frame)
    cv.imshow('colorized', colorized_frame)
