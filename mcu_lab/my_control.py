

import torch
from mcu_lab.stopWatch import stopWatch as watch


class controlSystem(watch):
    def __init__(self) -> None:
        super().__init__()
        self._round = 0
        self._runStatues={'select':0,'tracing':1}
        self._statue=self._runStatues['select']
        self._targPerson = torch.Tensor(1, 6).to(device='cuda:0',dtype=float)

        self.targPos = torch.tensor([0,0],device='cuda:0',dtype=float)
        return

    def term_start(self):  # 周期开始
        return super().start()

    def term_end(self):  # 周期结束
        return super().stop()

    @property
    def timeLastTerm(self):  # 上个周期的执行时间
        self._T = float(super().passTime) 
        return self._T
    def _sort_odds(tensor):#依据概率排序
        # 获取概率数据
        column = tensor[:, 4]
        # 对column进行排序，返回排序后的元素值和对应的索引
        sorted_column, sorted_indices = torch.sort(column)
        # 根据排序后的索引重新排列tensor的行
        sorted_tensor = tensor[sorted_indices, :]
        return sorted_tensor
    
    def Filter(self, manSites: torch.Tensor):  # 识别结果滤波器
        manList=controlSystem._sort_odds(manSites)
        if self._statue==0:
            return self._select_targ(manList)
        elif self._statue==1:
            return self._trace_targ(manList)
        else:
            raise Exception("风扇控制系统状态字出错，不是0 ro 1")
    
    def _select_targ(self,manlist:torch.Tensor):#选择目标
        self._statue=1
        #使用检测长宽和置信度综合判断
        #权值向量
        powVector=torch.tensor([1,1,0.2],device='cuda:0')
        #判断矩阵
        valueMat=torch.mul(manlist[:,2:5],powVector.t())
        # 计算每一行向量的平方和
        valueVect = torch.sum(valueMat**2, dim=1)
        # 获取最大值在valueVect中的位置
        max_idx = torch.argmax(valueVect) 
        self._targPerson= manlist[max_idx]      
        return self._targPerson
    def _trace_targ(self,manlist:torch.Tensor):#跟踪目标
        #使用检测5维度信息综合判断
        #权值向量
        powVector=torch.tensor([1,1,1,1,0.2],device='cuda:0')
        #偏差矩阵
        diffMat=(manlist-self._targPerson)[:,0:5]
        #判断矩阵
        valueMat=torch.mul(diffMat,powVector.t())
        # 计算每一行向量的平方和
        valueVect = torch.sum(valueMat**2, dim=1)
        # 获取min在valueVect中的位置
        min_idx = torch.argmin(valueVect)
        self._targPerson=manlist[min_idx]
        return self._targPerson
    def Controller(self, targMan: torch.Tensor):  # 控制器,一阶惯性
        #惯性系数
        m=1.0
        if self._T>m:#
            self.targPos=targMan[0:2]
        else:
            d_pos=targMan[0:2]-self.targPos
            d_targ=self._T/m*(d_pos)
            self.targPos+=d_targ
        return self.targPos
    pass

# import torch
# valueMat=torch.tensor([[1, 1, 0.5]])
# valueVect = torch.sum(valueMat**2, dim=1)
# valueVect