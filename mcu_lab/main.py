'''
Author: willing willing@123.com
Date: 2023-03-08 11:13:44
LastEditors: willing willing@123.com
LastEditTime: 2023-03-08 11:40:20
FilePath: \pytorch\main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import time
import threading
import serial
import my_camera
import cv2
import My_yolo
import torch
import My_serial
import sys
from time import sleep
from stopWatch import stopWatch as watch
from my_control import controlSystem


def camera_demo():
    cam0 = my_camera.My_camera()
    # while (1):
    # if cv2.waitKey(1) & 0xFF == ord('q'):  # 如果按下q 就截图保存并退出
    #     cv2.imwrite("test.png", frame)  # 保存路径
    #     break
    while (1):
        cam0.getFrame()
        cam0.imshow()
        pass
    return


def yolo_demo():
    yolov3 = My_yolo.Yolo_moddle()
    yolov3.inference(".\yolov3\img\zidane.jpg")
    yolov3.results_printTab()
    return


def yolo_camera_mix_demo():
    yolov3 = My_yolo.Yolo_moddle()
    cam0 = my_camera.My_camera()
    while(1):
        yolov3.inference(cam0.getFrame())
        yolov3.results_printTab()
        print(yolov3.result_only_person())
        yolov3.result_pic_man()
        yolov3.results_render()
        cam0.imshow()
        pass
    return


'''
串口与命令行透传函数
【bug】
1、input函数在无内容输入时会一直等待。
2、串口传输需要时间有时读取不完整。
经过测试
一号电机（m，俯仰）：调节角度0-100
二号电机（n，航向）：调节角度0-180
'''


def serial_demo():
    # 开启串口
    try:
        # 设置串口参数
        ser = serial.Serial('COM3', 115200, timeout=1)
    except:  # 没有默认串口则用户选择
        if not My_serial.list_available_ports():
            print('please connect the fan board and try again')
            return
        else:
            ser = serial.Serial(
                input('please select a com\n'), 115200, timeout=1)
    finally:  # 打印串口信息
        print("Connected to: " + ser.portstr)

        # 开始数据传输
    while True:
        # 从串口读取数据
        data = ser.readall().decode()
        if data:
            print('get a data:')
            print(data)
        # 从命令行读取输入并将其发送到串口
        if sys.stdin.readable():
            cmd = input('please input data\n').encode()
            if cmd:
                print('send a data:', cmd)
                ser.write(cmd)
                ser.flush()
                sleep(0.5)
                pass
            pass
    return


'''
单片机系统控制函数
经过测试
一号电机（m，俯仰）：调节角度0-100
二号电机（n，航向）：调节角度0-180
后面两个wait函数是其辅助函数
'''


def board_control_demo():
    # 初始化
    fanSystem = controlSystem()
    # 开启串口
    try:
        # 设置串口参数
        ser = serial.Serial('COM3', 115200, timeout=0.01)
    except:  # 没有默认串口则用户选择
        if not My_serial.list_available_ports():
            print('please connect the fan board and try again')
            raise My_serial.NoAvailablePortErrer('please connect the fan board and try again')
            return
        else:
            ser = serial.Serial(
                input('please select a com\n'), 115200, timeout=0.01)
    finally:  # 打印串口信息
        print("Connected to: " + ser.portstr)
    # 载入模型
    yolov3 = My_yolo.Yolo_moddle()
    # 开启摄像头
    cam0 = my_camera.My_camera()

    while True:  # 主循环
        fanSystem.timeLastTerm
        fanSystem.term_start()
        # 从串口读取数据
        print('get a data:', My_serial.serial_recive(ser))
        # yolo识别，并处理数据
        yolov3.inference(cam0.getFrame())
        personDiraction = yolov3.preProsess_result()
        
        try:
            targetPerson = fanSystem.Filter(personDiraction)
            # 在跟踪位置画点
            My_yolo.drow_point(cam0.frame, targetPerson[0:2],(0,0,255))
            targSite = fanSystem.Controller(targetPerson)
            # 在控制位置画点
            My_yolo.drow_point(cam0.frame, targSite)
        except IndexError:
            print('no person recognized')
            pass
        finally:
            cam0.imshow()
        My_serial.send_targPozition(ser, targSite)
        fanSystem.term_end()
    return


def main():
    # input_with_timeout("Please enter your name: ", 5)
    # try:
    #     name = input_with_timeout("Please enter your name: ", 5)  # 输入超时时间为5秒
    #     print("Hello, {}!".format(name))
    # except InputTimeoutError:
    #     print("Input timed out. Try again later.")

    yolo_camera_mix_demo()
    # board_control_demo()
    return


if __name__ == '__main__':
    main()


'''
实数重映射函数将data，在旧区间的相对位置映射成新区间内同位置的实数
'''


def reRange(data, rang_downOld, range_upOld, range_downNew, range_upNew):
    return (range_upNew-range_downNew)/(range_upOld-rang_downOld)*(data-rang_downOld)+range_downNew


def reRange_demo():
    a = reRange(55, 0, 180, 10, 170)
    print('rerange answer is {:.2f}'.format(a))
    return


'''可变参数函数示例'''


def testFunction(*args: object, **kwargs: object) -> int:
    print(args)
    print(kwargs)
    return 0


'''带定时中断的input函数，但是好像没用'''


class InputTimeoutError(Exception):
    pass


def input_with_timeout(prompt, timeout):
    print(prompt, end='', flush=True)
    timer = threading.Timer(timeout, lambda: print())  # 定时器，在超时时自动输出换行符
    timer.start()
    try:
        result = input()
        return result
    except:
        raise InputTimeoutError(
            "Input timed out after {} seconds".format(timeout))
        return
    finally:
        timer.cancel()  # 取消定时器
        return
