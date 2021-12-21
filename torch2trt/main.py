import os
import sys
import cv2
import copy
import torch
import argparse
root_path=os.path.dirname(os.path.abspath(os.path.dirname(__file__))) # 项目根路径：获取当前路径，再上级路径
sys.path.append(root_path)  # 将项目根路径写入系统路径
from utils.general import check_img_size,non_max_suppression_face,scale_coords,xyxy2xywh
from utils.datasets import letterbox
from detect_face import scale_coords_landmarks,show_results
from torch2trt.trt_model import TrtModel
cur_path=os.path.abspath(os.path.dirname(__file__))
def img_process(img_path,long_side=640,stride_max=32):
    '''
    图像预处理
    '''
    orgimg=cv2.imread(img_path)
    img0 = copy.deepcopy(orgimg)
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = long_side/ max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(long_side, s=stride_max)  # check img_size

    img = letterbox(img0, new_shape=imgsz,auto=False)[0] # auto True最小矩形   False固定尺度
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416
    img = torch.from_numpy(img)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img,orgimg

def img_vis(img,orgimg,pred,vis_thres = 0.6):
    '''
    预测可视化
    vis_thres: 可视化阈值
    '''

    print('img.shape: ', img.shape)
    print('orgimg.shape: ', orgimg.shape)

    no_vis_nums=0
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]  # normalization gain landmarks
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()

            for j in range(det.size()[0]):
                
                
                if det[j, 4].cpu().numpy() < vis_thres:
                    no_vis_nums+=1
                    continue

                xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                class_num = det[j, 15].cpu().numpy()
                orgimg = show_results(orgimg, xywh, conf, landmarks, class_num)

    cv2.imwrite(cur_path+'/result.jpg', orgimg)
    print('result save in '+cur_path+'/result.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default=cur_path+"/sample.jpg", help='img path') 
    parser.add_argument('--trt_path', type=str, required=True, help='trt_path') 
    parser.add_argument('--output_shape', type=list, default=[1,25200,16], help='input[1,3,640,640] ->  output[1,25200,16]') 
    opt = parser.parse_args()


    img,orgimg=img_process(opt.img_path) 
    model=TrtModel(opt.trt_path)
    pred=model(img.numpy()).reshape(opt.output_shape) # forward
    model.destroy()

    # Apply NMS
    pred = non_max_suppression_face(torch.from_numpy(pred), conf_thres=0.3, iou_thres=0.5)
   
    # ============可视化================
    img_vis(img,orgimg,pred)


