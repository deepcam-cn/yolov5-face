## 人脸追踪智能风扇
这是一个人脸追踪风扇的PC控制部分。实现了人脸识别并追踪。

## 用法
下载模型权重文件后

model_dir='best.pt'
标明权重文件位置

ser_send = False
Tcp_send = True
选择控制命令传送方式（双false关闭控制功能）

运行infer.py脚本，会将识别结果和追踪点标出。

## 本fork工作
PC控制器脚本：infer.py

此脚本实现实时图像采集、目标识别、基于识别结果计算控制变量、将控制目标展示与传输给风扇执行。
前两大功能基于原有脚本detect_face.py修改而来。后两个大功能基于mcu_lab中的多个库实现。包括控制算法my_control.py、TCP和串口通信方式的实现My_serial.py，my_tcpServer.py、离散控制所需要的计时功能stopWatch.py。

其中my_control.py中定义的两个滤波器使得对目标的追踪可以十分稳定。

注：使用onnx模型推理需要onnxruntime python库

风扇执行器系统设计：
https://github.com/gb16001/STM32_Pan-tilt_fan


## 参考：
fork from:
[yolov5face](https://github.com/gb16001/yolov5-face/tree/feature_out-detect)

dataset:[wider face](https://huggingface.co/datasets/wider_face)

annotation files:
[Jiankang Deng, Jia Guo, Yuxiang Zhou, Jinke Yu, Irene Kotsia, Stefanos Zafeiriou. Retinaface: Single-stage dense face localisation in the wild.[J]. arXiv preprint,arXiv:1905.00641,2019. ](https://drive.google.com/file/d/1tU_IjyOwGQfGNUvZGwWWM4SwxKp2PUQ8/view)



