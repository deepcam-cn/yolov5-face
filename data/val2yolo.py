import os
import cv2
import numpy as np
import shutil
import sys
from tqdm import tqdm


def xywh2xxyy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return x1, x2, y1, y2


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def wider2face(root, phase='val', ignore_small=0):
    data = {}
    with open('{}/{}/label.txt'.format(root, phase), 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip()
            if '#' in line:
                path = '{}/{}/images/{}'.format(root, phase, line.split()[-1])
                img = cv2.imread(path)
                height, width, _ = img.shape
                data[path] = list()
            else:
                box = np.array(line.split()[0:4], dtype=np.float32)  # (x1,y1,w,h)
                if box[2] < ignore_small or box[3] < ignore_small:
                    continue
                box = convert((width, height), xywh2xxyy(box))
                label = '0 {} {} {} {} -1 -1 -1 -1 -1 -1 -1 -1 -1 -1'.format(round(box[0], 4), round(box[1], 4),
                                                                             round(box[2], 4), round(box[3], 4))
                data[path].append(label)
    return data


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Missing path to WIDERFACE folder.')
        print('Run command: python3 val2yolo.py /path/to/original/widerface [/path/to/save/widerface/val]')
        exit(1)
    elif len(sys.argv) > 3:
        print('Too many arguments were provided.')
        print('Run command: python3 val2yolo.py /path/to/original/widerface [/path/to/save/widerface/val]')
        exit(1)

    root_path = sys.argv[1]
    if not os.path.isfile(os.path.join(root_path, 'val', 'label.txt')):
        print('Missing label.txt file.')
        exit(1)

    if len(sys.argv) == 2:
        if not os.path.isdir('widerface'):
            os.mkdir('widerface')
        if not os.path.isdir('widerface/val'):
            os.mkdir('widerface/val')

        save_path = 'widerface/val'
    else:
        save_path = sys.argv[2]

    datas = wider2face(root_path, phase='val')
    for idx, data in enumerate(datas.keys()):
        pict_name = os.path.basename(data)
        out_img = f'{save_path}/{idx}.jpg'
        out_txt = f'{save_path}/{idx}.txt'
        shutil.copyfile(data, out_img)
        labels = datas[data]
        f = open(out_txt, 'w')
        for label in labels:
            f.write(label + '\n')
        f.close()
