import torch
import detect_face
import numpy as np
import cv2
import serial
from mcu_lab import My_serial
from mcu_lab import my_control
import mcu_lab.My_yolo as My_yolo


class yoloFace:
    def __init__(self, weight='best.pt', conf_thres=0.6, iou_thres=0.5) -> None:
        self.weight = weight
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.model = detect_face.load_model(weight, self.device)
        pass

    def infer(self, img_np) -> torch.Tensor:
        img = torch.from_numpy(img_np).to(self.device)
        img = img.float()
        img /= 255.0
        pred = self.model(img)[0]
        pred = detect_face.non_max_suppression_face(
            pred, self.conf_thres, self.iou_thres)
        self.pred = pred
        print(pred)
        return pred

    def __call__(self, img_np) -> torch.Tensor:
        # # method 1
        # model=torch.load(weight, device)
        # pred = model['model'](img)[0]
        img = torch.from_numpy(img_np).to(self.device)
        img = img.float()
        img /= 255.0
        pred = self.model(img)[0]
        pred = detect_face.non_max_suppression_face(
            pred, self.conf_thres, self.iou_thres)
        self.pred = pred
        return pred# tensor list: bbox(xyxy),conf,landmark(5*2),class

    def plot(self, img_np):
        # Process detections
        img_rgb = cv2.cvtColor(img_np[0].transpose(
            (1, 2, 0)), cv2.COLOR_BGR2RGB)
        for i, det in enumerate(self.pred):  # detections per image
            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = det[j, 5:15].view(-1).tolist()
                class_num = det[j, 15].cpu().numpy()
                img_rgb = detect_face.show_results(
                    img_rgb, xyxy, conf, landmarks, class_num)
        return img_rgb


def open_serial(com='COM3', board=115200, timeout=0.01):
    try:
        # 设置串口参数
        ser = serial.Serial(com, board, timeout=timeout)
    except:  # 没有默认串口则用户选择
        if not My_serial.list_available_ports():
            print('please connect the fan board and try again')
            raise My_serial.NoAvailablePortErrer(
                'please connect the fan board and try again')

        else:
            ser = serial.Serial(
                input('please select a com\n'), board, timeout=timeout)
    finally:  # 打印串口信息
        print("Connected to: " + ser.portstr)
        return ser


if __name__ == '__main__':
    # mcu initial
    # open serial
    ser = open_serial()
    # init control system
    fanSystem = my_control.controlSystem()
    # 推理数据准备
    source = '0'  # 摄像头
    img_size = (480, 640)
    dataset = detect_face.LoadStreams(source, img_size)
    # 载入模型
    faceDetect = yoloFace(weight='weights/official_pretrained/yolov5n-face.pt')#'best.pt'
    # 对数据流中的数据推理
    for path, img_np, im0s, vid_cap in dataset:
        # record term time
        fanSystem.timeLastTerm
        fanSystem.term_start()
        # 从串口读取数据
        print('get a data:', My_serial.serial_recive(ser))
        detectXYxy = faceDetect(img_np)[0][:, [0, 1, 2, 3, 4, 15]]
        personDiraction = detectXYxy
        personDiraction[:, [0, 1]] = (
            detectXYxy[:, [0, 1]]+detectXYxy[:, [2, 3]])/2
        personDiraction[:, [2, 3]] = detectXYxy[:,
                                                [2, 3]]-detectXYxy[:, [0, 1]]
        personDiraction[:, [0, 2]] /= img_size[1]
        personDiraction[:, [1, 3]] /= img_size[0]
        print('detect:\n', personDiraction)
        img_rgb = faceDetect.plot(img_np)
        try:
            targetPerson = fanSystem.Filter(personDiraction)
            # 在跟踪位置画点
            My_yolo.drow_point(img_rgb, targetPerson[0:2], (0, 0, 255))
            targSite = fanSystem.Controller(targetPerson)
            # 在控制位置画点
            My_yolo.drow_point(img_rgb, targSite)
            My_serial.send_targPozition(ser, [1.0-targSite[0], targSite[1]])
        except IndexError:
            print('no person recognized')
            pass
        finally:
            # 摄像头是和人对立的，将图像左右调换回来正常显示
            cv2.imshow('result', img_rgb)#cv2.flip(img_rgb, 1)
            k = cv2.waitKey(1)
        fanSystem.term_end()
