## yolov5-face

在yolov5的基础上增加landmark预测分支，loss使用wingloss,使用yolov5s取得了相对于retinaface-r50更好的性能。

#### WiderFace测试

* 在wider face val精度（单尺度最大边输入分辨率：**640**）

| Method              | Backbone  | Easy  | Medium | Hard  | \#Params(M) | \#Flops(G) |
  | ------------------- | --------- | ----- | ------ | ----- | ----------- | ---------- |
  | DSFD (CVPR19)       | ResNet152 | 94.29 | 91.47  | 71.39 | 120.06      | 259.55     |
  | RetinaFace (CVPR20) | ResNet50  | 94.92 | 91.90  | 64.17 | 29.50       | 37.59      |
  | HAMBox (CVPR20)     | ResNet50  | 95.27 | 93.76  | 76.75 | 30.24       | 43.28      |
  | TinaFace (Arxiv20)  | ResNet50  | 95.61 | 94.25  | 81.43 | 37.98       | 172.95     |
  | -                   | -         | -     | -      | -     | -           | -          |
  | yolov5s6            |           | 95.48 | 93.66  | 82.8  | 12.402      | 8.414      |
  | yolov5m6            |           | 95.66 | 94.1   | 85.2  | 35.519      | 25.788     |
  | yolov5m6+           |           | 96.06 | 94.4   | 85.4  | 63.077      | 45.520     |

#### 模型测试下载地址

* yolov5s:链接: https://pan.baidu.com/s/1t51CFeofy1slOw_lgb3UDg  密码: mkh0
* Yolov5m:

#### 模型测试效果

![](data/images/result.jpg)



#### References

https://github.com/ultralytics/yolov5

https://github.com/DayBreak-u/yolo-face-with-landmark

https://github.com/xialuxi/yolov5_face_landmark

https://github.com/biubug6/Pytorch_Retinaface

https://github.com/deepinsight/insightface
