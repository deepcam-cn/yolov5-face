## What's New

**2021.12**: Yolov5-face to TensorRT. 

|   Backbone   | Pytorch(ms) | TensorRT_FP16(ms) |
| :----------: | :---------: | :---------------: |
| yolov5n-0.5  |     7.7     |        2.1        |
| yolov5n-face |     7.7     |        2.4        |
| yolov5s-face |     5.6     |        2.2        |
| yolov5m-face |     9.9     |        3.3        |
| yolov5l-face |    15.9     |        4.5        |

> Pytorch=1.10.0+cu102    TensorRT=8.2.0.6   Hardware=rtx2080ti

**2021.11**: BlazeFace

| Method               | multi scale | Easy  | Medium | Hard  | Model Size(MB) | Link  |
| -------------------- | ----------- | ----- | ------ | ----- | -------------- | ----- |
| BlazeFace            | Ture        | 88.5  | 85.5   | 73.1  | 0.472          | https://github.com/PaddlePaddle/PaddleDetection |
| BlazeFace-FPN-SSH    | Ture        | 90.7  | 88.3   | 79.3  | 0.479          | https://github.com/PaddlePaddle/PaddleDetection |
| yolov5-blazeface     | True        | 90.4  | 88.7   | 78.0  | 0.493          | https://pan.baidu.com/s/1RHp8wa615OuDVhsO-qrMpQ pwd:r3v3 |
| yolov5-blazeface-fpn | True        | 90.8  | 89.4   | 79.1  | 0.493          |  -    |


**2021.08**: Add new training dataset [Multi-Task-Facial](https://drive.google.com/file/d/1Pwd6ga06cDjeOX20RSC1KWiT888Q9IpM/view?usp=sharing),improve large face detection.
| Method               | Easy  | Medium | Hard  | 
| -------------------- | ----- | ------ | ----- |
| ***YOLOv5s***        | 94.56 | 92.92  | 83.84 |
| ***YOLOv5m***        | 95.46 | 93.87  | 85.54 |


## Introduction

Yolov5-face is a real-time,high accuracy face detection.

![](data/images/yolov5-face-p6.png)

## Performance

Single Scale Inference on VGA resolution（max side is equal to 640 and scale).

***Large family***

| Method              | Backbone       | Easy  | Medium | Hard  | \#Params(M) | \#Flops(G) |
| :------------------ | -------------- | ----- | ------ | ----- | ----------- | ---------- |
| DSFD (CVPR19)       | ResNet152      | 94.29 | 91.47  | 71.39 | 120.06      | 259.55     |
| RetinaFace (CVPR20) | ResNet50       | 94.92 | 91.90  | 64.17 | 29.50       | 37.59      |
| HAMBox (CVPR20)     | ResNet50       | 95.27 | 93.76  | 76.75 | 30.24       | 43.28      |
| TinaFace (Arxiv20)  | ResNet50       | 95.61 | 94.25  | 81.43 | 37.98       | 172.95     |
| SCRFD-34GF(Arxiv21) | Bottleneck Res | 96.06 | 94.92  | 85.29 | 9.80        | 34.13      |
| SCRFD-10GF(Arxiv21) | Basic Res      | 95.16 | 93.87  | 83.05 | 3.86        | 9.98       |
| -                   | -              | -     | -      | -     | -           | -          |
| ***YOLOv5s***       | CSPNet         | 94.67 | 92.75  | 83.03 | 7.075       | 5.751      |
| **YOLOv5s6**        | CSPNet         | 95.48 | 93.66  | 82.8  | 12.386      | 6.280      |
| ***YOLOv5m***       | CSPNet         | 95.30 | 93.76  | 85.28 | 21.063      | 18.146     |
| **YOLOv5m6**        | CSPNet         | 95.66 | 94.1   | 85.2  | 35.485      | 19.773     |
| ***YOLOv5l***       | CSPNet         | 95.78 | 94.30  | 86.13 | 46.627      | 41.607     |
| ***YOLOv5l6***      | CSPNet         | 96.38 | 94.90  | 85.88 | 76.674      | 45.279     |


***Small family***

| Method               | Backbone        | Easy  | Medium | Hard  | \#Params(M) | \#Flops(G) |
| -------------------- | --------------- | ----- | ------ | ----- | ----------- | ---------- |
| RetinaFace (CVPR20   | MobileNet0.25   | 87.78 | 81.16  | 47.32 | 0.44        | 0.802      |
| FaceBoxes (IJCB17)   |                 | 76.17 | 57.17  | 24.18 | 1.01        | 0.275      |
| SCRFD-0.5GF(Arxiv21) | Depth-wise Conv | 90.57 | 88.12  | 68.51 | 0.57        | 0.508      |
| SCRFD-2.5GF(Arxiv21) | Basic Res       | 93.78 | 92.16  | 77.87 | 0.67        | 2.53       |
| -                    | -               | -     | -      | -     | -           | -          |
| ***YOLOv5n***        | ShuffleNetv2    | 93.74 | 91.54  | 80.32 | 1.726       | 2.111      |
| ***YOLOv5n-0.5***    | ShuffleNetv2    | 90.76 | 88.12  | 73.82 | 0.447       | 0.571      |



## Pretrained-Models

| Name        | Easy  | Medium | Hard  | FLOPs(G) | Params(M) | Link                                                         |
| ----------- | ----- | ------ | ----- | -------- | --------- | ------------------------------------------------------------ |
| yolov5n-0.5 | 90.76 | 88.12  | 73.82 | 0.571    | 0.447     | Link: https://pan.baidu.com/s/1UgiKwzFq5NXI2y-Zui1kiA  pwd: s5ow, https://drive.google.com/file/d/1XJ8w55Y9Po7Y5WP4X1Kg1a77ok2tL_KY/view?usp=sharing |
| yolov5n     | 93.61 | 91.52  | 80.53 | 2.111    | 1.726     | Link: https://pan.baidu.com/s/1xsYns6cyB84aPDgXB7sNDQ  pwd: lw9j,https://drive.google.com/file/d/18oenL6tjFkdR1f5IgpYeQfDFqU4w3jEr/view?usp=sharing |
| yolov5s     | 94.33 | 92.61  | 83.15 | 5.751    | 7.075     | Link: https://pan.baidu.com/s/1fyzLxZYx7Ja1_PCIWRhxbw  Link: eq0q,https://drive.google.com/file/d/1zxaHeLDyID9YU4-hqK7KNepXIwbTkRIO/view?usp=sharing |
| yolov5m     | 95.30 | 93.76  | 85.28 | 18.146   | 21.063    | Link: https://pan.baidu.com/s/1oePvd2K6R4-gT0g7EERmdQ  pwd: jmtk, https://drive.google.com/file/d/1Sx-KEGXSxvPMS35JhzQKeRBiqC98VDDI |
| yolov5l     | 95.78 | 94.30  | 86.13 | 41.607   | 46.627    | Link: https://pan.baidu.com/s/11l4qSEgA2-c7e8lpRt8iFw  pwd: 0mq7, https://drive.google.com/file/d/16F-3AjdQBn9p3nMhStUxfDNAE_1bOF_r |

## Data preparation

1. Download WIDERFace datasets.
2. Download annotation files from [google drive](https://drive.google.com/file/d/1tU_IjyOwGQfGNUvZGwWWM4SwxKp2PUQ8/view?usp=sharing).

```shell
python3 train2yolo.py
python3 val2yolo.py
```



## Training

```shell
CUDA_VISIBLE_DEVICES="0,1,2,3" python3 train.py --data data/widerface.yaml --cfg models/yolov5s.yaml --weights 'pretrained models'
```



## WIDERFace Evaluation

```shell
python3 test_widerface.py --weights 'your test model' --img-size 640

cd widerface_evaluate
python3 evaluation.py
```

#### Test

![](data/images/result.jpg)


#### Android demo

https://github.com/FeiGeChuanShu/ncnn_Android_face/tree/main/ncnn-android-yolov5_face

#### opencv dnn demo

https://github.com/hpc203/yolov5-face-landmarks-opencv-v2

#### References

https://github.com/ultralytics/yolov5

https://github.com/DayBreak-u/yolo-face-with-landmark

https://github.com/xialuxi/yolov5_face_landmark

https://github.com/biubug6/Pytorch_Retinaface

https://github.com/deepinsight/insightface


#### Citation 
- If you think this work is useful for you, please cite 

      @article{YOLO5Face,
      title = {YOLO5Face: Why Reinventing a Face Detector},
      author = {Delong Qi and Weijun Tan and Qi Yao and Jingfeng Liu},
      booktitle = {ArXiv preprint ArXiv:2105.12931},
      year = {2021}
      }

#### Main Contributors
https://github.com/derronqi  

https://github.com/changhy666 

https://github.com/bobo0810 

