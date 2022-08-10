import time

import cv2
import numpy as np
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import time_sync

if __name__ == "__main__":
    start_time = time_sync()

    use_gpu = torch.cuda.is_available()

    weights = 'yolov5s.pt'
    w = str(weights[0] if isinstance(weights, list) else weights)

    # torch.device('cuda:0')
    model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights)  # 加载模型
    if use_gpu:
        model.cuda()

    height, width = 640, 640

    cost = time_sync() - start_time
    print(cost)

    img0 = cv2.imread('data/images/bus.jpg')
    img = cv2.resize(img0, (height, width))  # 尺寸变换
    img = img / 255.
    img = img[:, :, ::-1].transpose((2, 0, 1))  # HWC转CHW
    img = np.expand_dims(img, axis=0)  # 扩展维度至[1,3,640,640]
    img = torch.from_numpy(img.copy())  # numpy转tensor
    if use_gpu:
        img = img.cuda()
    img = img.to(torch.float32)  # float64转换float32
    pred = model(img, augment='store_true', visualize='store_true')[0]

    if use_gpu:
        pred = pred.cpu()
    pred.clone().detach()
    pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)  # 非极大值抑制
    names = model.names
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                print('{},{},{},{}'.format(xyxy, conf.numpy(), cls.numpy(),
                                           names[int(cls)]))  # 输出结果：xyxy检测框左上角和右下角坐标，conf置信度，cls分类结果
                img0 = cv2.rectangle(img0, (int(xyxy[0].numpy()), int(xyxy[1].numpy())),
                                     (int(xyxy[2].numpy()), int(xyxy[3].numpy())), (0, 255, 0), 2)
    cost = time_sync() - start_time
    print(cost)
    cv2.imwrite('out.jpg', img0)  # 简单画个框
