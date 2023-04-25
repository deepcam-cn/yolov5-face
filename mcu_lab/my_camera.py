'''
Author: willing willing@123.com
Date: 2023-03-08 10:46:16
LastEditors: willing willing@123.com
LastEditTime: 2023-03-08 10:48:13
FilePath: \pytorch\my_camera.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import cv2


class My_camera:
    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 打开摄像头。第一个参数为1则是外接摄像头
        pass

    def getFrame(self):      # get a frame
        ret, frame = self.cap.read()  # 摄像头读取,ret为是否成功打开摄像头,true,false。 frame为视频的每一帧图像
        self.frame = cv2.flip(frame,1)  # 摄像头是和人对立的，将图像左右调换回来正常显示
        
        return  self.frame
    pass

    def imshow(self):
        # show a frame
        cv2.imshow("vedio", self.frame)  # 生成摄像头窗口
        cv2.waitKey(1)        # 加入延时，让显示来得及
    pass

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()
        pass


pass


# while (1):
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # 如果按下q 就截图保存并退出
#         cv2.imwrite("test.png", frame)  # 保存路径
#         break
