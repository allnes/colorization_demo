from openvino.inference_engine import IENetwork, IECore
import cv2 as cv
import numpy as np

net = IENetwork(model="colorization.xml", weights="colorization.bin")
ie = IECore()
exec_net = ie.load_network(network=net, device_name="CPU")

input_blob = next(iter(net.inputs))
n, c, h_in, w_in = net.inputs[input_blob].shape
images = np.ndarray(shape=(n, c, h_in, w_in))

if 1:
    cap = cv.VideoCapture('test.mp4')
else:
    cap = cv.VideoCapture(0)

while cv.waitKey(1) < 0:
    hasFrame, original_frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    # preprocessing frame
    frame = cv.cvtColor(cv.cvtColor(original_frame, cv.COLOR_RGB2GRAY), cv.COLOR_GRAY2RGB)
    img_rgb = (frame[:,:,[2, 1, 0]] * 1.0 / 255).astype(np.float32)
    img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
    img_l = img_lab[:,:,0] # pull out L channel
    (H_orig,W_orig) = img_rgb.shape[:2] # original image size

    img_rs = cv.resize(img_rgb, (w_in, h_in))
    img_lab_rs = cv.cvtColor(img_rs, cv.COLOR_RGB2Lab)
    img_l_rs = img_lab_rs[:,:,0]
    img_l_rs -= 50

    for i in range(n):
        images[i] = img_l_rs

    # network inference
    res = exec_net.infer(inputs={input_blob: images})

    # get result
    out = res['class8_ab'][0,:,:,:].transpose((1,2,0))
    out = cv.resize(out, (W_orig, H_orig))
    img_lab_out = np.concatenate((img_l[:,:,np.newaxis],out),axis=2)
    img_bgr_out = np.clip(cv.cvtColor(img_lab_out, cv.COLOR_Lab2BGR), 0, 1)

    # show result
    imshowSize = (640, 480)
    frame = cv.resize(frame, imshowSize)
    original_frame = cv.resize(original_frame, imshowSize)
    cv.imshow('origin', original_frame)
    cv.imshow('gray', cv.cvtColor(frame, cv.COLOR_RGB2GRAY))
    cv.imshow('colorized', cv.resize(img_bgr_out, imshowSize))
