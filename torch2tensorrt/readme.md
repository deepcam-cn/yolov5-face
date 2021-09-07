English | [简体中文](readme_CN.md)



# Overall process

## 1.Pytorch->ONNX

`/models/export.py` configure `weights` ，export **onnx file**

## 2.TensorRT inference
`/torch2tensorrt/main.py`  configure **onnx file path**

Image preprocessing -> TensorRT inference -> visualization 



# Time-consuming comparison

| |Pytorch |TensorRT_FP16 |
|:---:|:----:|:----:|
|yolov5n-0.5|11.9ms|2.9ms|
|yolov5n-face|20.7ms|2.5ms|
|yolov5s-face|25.2ms|3.0ms|
|yolov5m-face|61.2ms|3.0ms|
|yolov5l-face|109.6ms|3.6ms|
> Note: (1) Model inference  (2) Resolution 640x640 (3)TensorRT7.2.2-1 cuda11.1 （4）Average time spent in 100 rounds after warm-up



# Visualization

<table>
    <tr>
            <th>yolov5n-0.5</th>
            <th>yolov5n-face</th>
    </tr>
    <tr>
        <td><img src="./imgs/yolov5n-0.5.jpg" /></td>
        <td><img src="./imgs/yolov5n-face.jpg" /></td>
    </tr>
</table>

<table>
    <tr>
            <th>yolov5s-face</th>
            <th>yolov5m-face</th>
            <th>yolov5l-face</th>
    </tr>
    <tr>
        <td><img src="./imgs/yolov5s-face.jpg" /></td>
        <td><img src="./imgs/yolov5m-face.jpg" /></td>
        <td><img src="./imgs/yolov5l-face.jpg" /></td>
    </tr>
</table>


# TODO

- [ ] TensorRT support dynamic resolution

- [ ] Pytorch -> ONNX -> NCNN 

- [ ] Pytorch -> ONNX -> MNN

