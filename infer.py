import torch
import detect_face
import numpy as np
import cv2


if __name__ == '__main__':
    weight = 'best.pt'
    source = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf_thres = 0.6
    iou_thres = 0.5
    # model = detect_face.load_model(weight, device)
    # infer_result= detect_face.detect(model,source,device,'runs/detect','exp',False,False,True)
    # print(infer_result)

    dataset = detect_face.LoadStreams(source, img_size=(640, 640))
    model = detect_face.load_model(weight, device)
    for path, img_np, im0s, vid_cap in dataset:
        # # method 1
        # model=torch.load(weight, device)
        # pred = model['model'](img)[0]

        img = torch.from_numpy(img_np).to(device)
        img = img.float()
        img /= 255.0
        pred = model(img)[0]
        pred = detect_face.non_max_suppression_face(
            pred, conf_thres, iou_thres)
        print(pred)

        # Process detections
        img_rgb = cv2.cvtColor(img_np[0].transpose(
            (1, 2, 0)), cv2.COLOR_BGR2RGB)
        for i, det in enumerate(pred):  # detections per image
            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = det[j, 5:15].view(-1).tolist()
                class_num = det[j, 15].cpu().numpy()
                img_rgb = detect_face.show_results(
                    img_rgb, xyxy, conf, landmarks, class_num)
        cv2.imshow('result', img_rgb)
        k = cv2.waitKey(1)
