import torch
import detect_face
import numpy as np
import cv2


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
        return pred

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


if __name__ == '__main__':
    source = '0'#摄像头
    img_size = (480, 640)
    # 推理数据准备
    dataset = detect_face.LoadStreams(source, img_size)
    # 载入模型
    faceDetect = yoloFace()
    # 对数据流中的数据推理
    for path, img_np, im0s, vid_cap in dataset:
        faceDetect.infer(img_np)
        img_rgb = faceDetect.plot(img_np)
        cv2.imshow('result', img_rgb)
        k = cv2.waitKey(1)
