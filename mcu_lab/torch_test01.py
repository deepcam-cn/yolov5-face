'''
Author: willing willing@123.com
Date: 2023-03-07 20:35:21
LastEditors: willing willing@123.com
LastEditTime: 2023-03-08 10:41:49
FilePath: \pytorch\torch_test01.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch

# Model
model = torch.hub.load("./yolov3",'yolov5l','--weights=.\yolov3\weight\yolov5n.pt',source='local')  # or yolov5n - yolov5x6, custom
# model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5m-seg.pt') 
# load from PyTorch Hub (WARNING: inference not yet supported)
# 这里'yolov5m-seg.pt'参数是模型的权重文件，实测没什么用

# Images
img = ".\yolov3\img\zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.show()  # or .print(), .show(), .save(), .crop(), .pandas(), etc.

# print(results.tolist())

# need_xywh=results.xywhn[0]
# need_xywh.to("cpu")
# print(need_xywh) # xywhn=[xywh,置信度,类别编号],x从左到右;y从上到下.数值为归一化值