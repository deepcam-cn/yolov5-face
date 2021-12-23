from models.experimental import attempt_load
from torch2trt.trt_model import TrtModel
import argparse
import torch
import time
from tqdm import tqdm


def run(model,img,warmup_iter,iter):
    
    
    print('start warm up...')
    for _ in tqdm(range(warmup_iter)):
        model(img) 
    
   
    print('start calculate...')
    torch.cuda.synchronize()
    start = time.time()
    for __ in tqdm(range(iter)):
        model(img) 
        torch.cuda.synchronize()
    end = time.time()
    return ((end - start) * 1000)/float(iter)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_path', type=str,required=True, help='torch weights path')  
    parser.add_argument('--trt_path', type=str,required=True, help='tensorrt weights path')

    parser.add_argument('--device', type=int,default=0, help='cuda device')
    parser.add_argument('--img_shape', type=list,default=[1,3,640,640], help='tensorrt weights path')
    parser.add_argument('--warmup_iter', type=int, default=100,help='warm up iter')  
    parser.add_argument('--iter', type=int, default=300,help='average elapsed time of iterations')  
    opt = parser.parse_args()


    # -----------------------torch-----------------------------------------
    img = torch.zeros(opt.img_shape)
    model = attempt_load(opt.torch_path, map_location=torch.device('cpu'))  # load FP32 model
    model.eval()
    total_time=run(model.to(opt.device),img.to(opt.device),opt.warmup_iter,opt.iter)
    print('Pytorch is  %.2f ms/img'%total_time)

    # -----------------------tensorrt-----------------------------------------
    model=TrtModel(opt.trt_path)
    total_time=run(model,img.numpy(),opt.warmup_iter,opt.iter)
    model.destroy()
    print('TensorRT is  %.2f ms/img'%total_time)
