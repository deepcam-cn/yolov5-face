
from time import time
class stopWatch:
    def __init__(self) -> None:
        self.start_time=self.stop_time=0
        pass
    def start(self):
        self.start_time=time()
        return self.start_time
    def stop(self):
        self.stop_time=time()
        return self.stop_time
    @property
    def passTime(self):
        elapsed_time = self.stop_time - self.start_time
        print("执行时间:", elapsed_time, "秒")
        return elapsed_time
    def refrash(self):
        self.start_time=0
        self.stop_time=0
        return