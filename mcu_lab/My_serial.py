import serial  # 导入串口通信库
from time import sleep
class NoAvailablePortErrer(Exception):
    pass

class my_Serial:
    # com口，波特率，比特大小，停止位,奇偶校验位
    def __init__(self, port, baudrate=115200, bytesize=8, stopbits=1, parity="N") -> None:
        self.port = port
        self.baudrate = baudrate
        self.bytesize = bytesize
        self.stopbits = stopbits
        self.parity = parity
        self.com = serial.Serial(
            port, baudrate, bytesize, parity, stopbits, parity=parity)
        self.com.open()
        if(ser.isOpen()):
            print("串口打开成功！")
        else:
            print("串口打开失败！")
        pass
    pass


def serial_recive(ser: serial.Serial):  # 串口接收数据
    try:
        if ser.in_waiting > 0:
            data = ser.read_all().decode()
            if data:
                return data
            else:
                return None
        else:
            return None
    except UnicodeDecodeError:
        return None
    


def wait_untill_ser_writable(ser: serial.Serial):  # 等待串口可写
    while(not ser.writable()):
        print('sleep once,wait serial to be writable')
        sleep(0.01)
    return


def wait_recive_ser_responce(ser: serial.Serial):  # 等待读取串口回应

    while(not ser.in_waiting > 0):  # 等待回应
        sleep(0.01)
        # print('recive sleep once ')
        pass
    try:
        data = ser.read_all().decode()  # 打印回应
        if data:
            print('get a data:')
            print(data, end='')
            pass
        return data
    except UnicodeDecodeError:
        # print('UnicodeDecodeError from serial')
        return None


def send_targPozition(ser: serial.Serial, targDiraction):  # 指引云台转向

    try:
        # 发送横纵坐标
        wait_untill_ser_writable(ser)
        ser.write(('n {x_deraction:d}\n'.format(
            x_deraction=int(targDiraction[0]*180))).encode())
        ser.write(('m {y_deraction:d}\n'.format(
            y_deraction=int(targDiraction[1]*100))).encode())
        wait_recive_ser_responce(ser)
    except IndexError:

        pass
    return


'''待填充的serial功能
ser.isOpen()：查看端口是否被打开。
ser.open() ：打开端口‘。
ser.close()：关闭端口。
ser.read()：从端口读字节数据。默认1个字节。
ser.read_all():从端口接收全部数据。
ser.write("hello")：向端口写数据。
ser.readline()：读一行数据。
ser.readlines()：读多行数据。
in_waiting()：返回接收缓存中的字节数。
flush()：等待所有数据写出。
flushInput()：丢弃接收缓存中的所有数据。
flushOutput()：终止当前写操作，并丢弃发送缓存中的数据。
'''


ser = serial.Serial()


def port_open_recv():  # 对串口的参数进行配置
    ser.port = 'com3'
    ser.baudrate = 115200
    ser.bytesize = 8
    ser.stopbits = 1
    ser.parity = "N"  # 奇偶校验位
    ser.open()
    if(ser.isOpen()):
        print("串口打开成功！")
    else:
        print("串口打开失败！")
# isOpen()函数来查看串口的开闭状态


def port_close():
    ser.close()
    if(ser.isOpen()):
        print("串口关闭失败！")
    else:
        print("串口关闭成功！")


def send(send_data):
    if(ser.isOpen()):
        ser.write(send_data.encode('utf-8'))  # 编码
        print("发送成功", send_data)
    else:
        print("发送失败！")


def list_available_ports():
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No COM ports found")
        return False
    else:
        print("Available COM ports:")
        for port in ports:
            print(port.device)
            pass
        return True


if __name__ == '__main__':
    import concurrent.futures
    # 创建 ThreadPoolExecutor 对象
    executor = concurrent.futures.ThreadPoolExecutor()
    port_open_recv()
    while True:
        a = input("输入要发送的数据：")
        ret=executor.submit(send,a)
        # send(a)
        sleep(0.5)  # 起到一个延时的效果，这里如果不加上一个while True，程序执行一次就自动跳出了
