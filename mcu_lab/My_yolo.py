import torch
import cv2
import onnx
import onnxruntime as ort
import numpy as np


def tensor_pic_man(tensor_2d: torch.Tensor) -> torch.Tensor:  # 输入结果的二维张量，选出为人的部分
    return tensor_2d[tensor_2d[:, -1] == 0, :]


def infer_result_filter():  # 对识别结果进行滤波（未完成）
    return


def drow_point(img, site: tuple, color=(0, 255, 0)):  # 在指定像素标一个圆(默认绿色)
    if not len(site) == 2:
        raise ValueError("site参数位置维数不匹配,需为2")
    real_site = (int(site[0]*img.shape[1]), int(site[1]*img.shape[0]))
    cv2.circle(img, real_site, 5, color, 2)
    return


class Yolo_moddle:
    def __init__(self, path="yolov5", modle='custom', model_path='yolov5n.pt') -> None:
        # or yolov5n - yolov5x6, custom
        # self.model = torch.hub.load(path, modle,model_path ,force_reload=True,source='local')
        self.model = ort.InferenceSession('self_model\epoch68/best02.onnx')
        self.modelPath = path
        self.modelName = modle
        pass

    def result_pic_man(self) -> torch.Tensor:  # 删除结果中不是人的识别内容
        self.inference_results.xywh[0] = tensor_pic_man(
            self.inference_results.xywh[0])
        self.inference_results.xywhn[0] = tensor_pic_man(
            self.inference_results.xywhn[0])
        self.inference_results.xyxy[0] = tensor_pic_man(
            self.inference_results.xyxy[0])
        self.inference_results.xyxyn[0] = tensor_pic_man(
            self.inference_results.xyxyn[0])
        return self.inference_results.xywhn[0]

    def result_only_person(self) -> torch.Tensor:  # 将结果中的人选出来作为张量输出
        tempBox = self.results_xywh()
        self.reultsPerson = tempBox[tempBox[:, -1] == 0, :]
        return self.reultsPerson

    def result_pickMan_way2(self) -> torch.Tensor:  # 提取人识别结果法2,not recomand

        for b in self.results_xywh():
            if b[-1] == 0:
                try:
                    returnBox = torch.cat((returnBox, b), dim=0)
                except:
                    returnBox = b
                print('man resualt :', b)
            pass
        try:
            return returnBox
        except:
            return None

    def inference(self, img):  # 推理,img= file, Path, PIL, OpenCV, numpy, list
        
        img_float = np.expand_dims(img.astype(np.float32), axis=0).transpose(0, 3, 1, 2)
        self.inference_results = self.model.run([self.model.get_outputs()[0].name], {'input': img_float})
        self.img = img
        return self.inference_results
        pass
    # 一下所有方法都写于/models/common.py     都通过_run()函数实现

    def preProsess_result(self):
        # yolov3.results_printTab()
        # print(personDiraction[:, 0:2])
        personDiraction = self.result_pic_man()
        self.results_render()
        return personDiraction

    def results_show(self):
        self.inference_results.show()
        pass

    # xywhn=[xywh,置信度,类别编号],x从左到右;y从上到下.数值为归一化值
    def results_xywh(self) -> list:
        return self.inference_results.xywhn[0]

    def results_printTab(self):  # 表格化输出
        print(self.inference_results.pandas().xyxy[0])
        pass

    def results_save(self):
        self.inference_results.save()
        return

    def results_crop(self):
        self.inference_results.crop()
        return

    def results_list(self):
        print(self.inference_results.tolist())
        return

    def results_print(self):  # 展示测试信息（不是测试结果）
        self.inference_results.print()
        return

    def results_render(self):  # 对原图进行标注，并排序测试到的信息（可以用来展示识别后的边框而不用保存）
        self.inference_results.render()
        return
