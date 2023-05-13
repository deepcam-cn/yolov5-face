import concurrent.futures
import torch
import detect_face
import numpy as np
import cv2
import serial
from mcu_lab import My_serial
from mcu_lab import my_control
from mcu_lab import my_tcpServer
import mcu_lab.My_yolo as My_yolo
import threading


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
        return pred  # tensor list: bbox(xyxy),conf,landmark(5*2),class

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
        print("Connected to: " + ser.portstr)  # 打印串口信息
        return ser
    except:  # 没有默认串口则用户选择
        if not My_serial.list_available_ports():
            print('please connect the fan board and try again')
            raise My_serial.NoAvailablePortErrer(
                'please connect the fan board and try again')
            return None
        else:
            ser = serial.Serial(
                input('please select a com\n'), board, timeout=timeout)
            print("Connected to: " + ser.portstr)  # 打印串口信息
            return ser


# 图像识别
def obj_detect():

    detectXYxy = faceDetect(img_np)[0][:, [0, 1, 2, 3, 4, 15]]
    personDiraction = detectXYxy
    personDiraction[:, [0, 1]] = (
        detectXYxy[:, [0, 1]]+detectXYxy[:, [2, 3]])/2
    personDiraction[:, [2, 3]] = detectXYxy[:,
                                            [2, 3]]-detectXYxy[:, [0, 1]]
    personDiraction[:, [0, 2]] /= img_size[1]
    personDiraction[:, [1, 3]] /= img_size[0]
    print('detect:\n', personDiraction)
    return personDiraction

# 结果展示和控制坐标发送


def img_draw_ser_send():
    try:
        img_rgb = faceDetect.plot(img_np)
        targetPerson = fanSystem.Filter(personDiraction)
        targSite = fanSystem.Controller(targetPerson)
        ser_send_future = executor.submit(My_serial.send_targPozition, ser, [
            1.0-targSite[0], targSite[1]])
        
        # 在跟踪位置画点
        My_yolo.drow_point(img_rgb, targetPerson[0:2], (0, 0, 255))
        # 在控制位置画点
        My_yolo.drow_point(img_rgb, targSite)
        poz_cmd=ser_send_future.result()
        my_TCP.write(poz_cmd)
        print(f'send tcp :{poz_cmd}')

    # My_serial.send_targPozition(ser, [1.0-targSite[0], targSite[1]])
    except IndexError:
        print('no person recognized')
        pass
    finally:
        # 摄像头是和人对立的，将图像左右调换回来正常显示
        cv2.imshow('result', img_rgb)  # cv2.flip(img_rgb, 1)
        k = cv2.waitKey(1)
    return ser_send_future.result()


if __name__ == '__main__':
    # mcu initial
    # 创建 ThreadPoolExecutor 对象
    executor = concurrent.futures.ThreadPoolExecutor()
    # 建立TCP连接
    my_TCP = my_tcpServer. oneTCPserver()
    executor.submit(my_TCP.loop)
    # open serial
    ser = open_serial()
    # init control system
    fanSystem = my_control.controlSystem()
    # 推理数据准备
    source = '0'  # 摄像头
    img_size = (480, 640)
    dataset = detect_face.LoadStreams(source, img_size)
    # 载入模型
    faceDetect = yoloFace(
        weight='weights/official_pretrained/yolov5n-0.5.pt')  # 'best.pt'
    # 对数据流中的数据推理
    for path, img_np, im0s, vid_cap in dataset:#dataset是一个迭代器
        # record term time
        fanSystem.timeLastTerm
        fanSystem.term_start()
        # object detect
        personDiraction = obj_detect()
        # 从串口读取数据
        try:
            show_future.result()  # 等待上轮显示、发送完毕
        except NameError:
            pass
        print('get a data:', My_serial.serial_recive(ser))
        # show img and send control command,run by thread pool
        show_future = executor.submit(img_draw_ser_send)
        show_future.priority = -1
        # timer stop
        fanSystem.term_end()
