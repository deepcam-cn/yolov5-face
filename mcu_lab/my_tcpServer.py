import io
import threading
import socket
import concurrent.futures
from time import sleep
HOST = '192.168.137.1'  # 服务器IP地址
PORT = 8888        # 服务器端口号


def single_server():

    # 创建一个TCP socket对象
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # 将socket对象绑定到指定IP地址和端口号
        s.bind((HOST, PORT))
        # 开启监听模式，等待客户端连接
        s.listen()
        print('等待客户端连接...')
        # 接受客户端连接，返回连接对象和客户端地址
        conn, addr = s.accept()
        print(f'已连接客户端 {addr}')

        # 接收来自客户端的数据，直到客户端关闭连接
        while True:
            data = conn.recv(1024)  # 每次最多接收1024字节数据
            if not data:
                break
            # 处理接收到的数据
            print(f'收到客户端消息：{data.decode()}')

        # 关闭连接
        conn.close()


'''
以下为多线程例子from chatgpt
'''


def handle_client(conn: socket.socket, addr):
    print(f"[NEW CONNECTION] {addr} connected.")

    connected = True
    while connected:
        # 接收客户端发送的消息
        try:
            message = conn.recv(1024)
            if message:
                print(f"[{addr}] {message.decode()}")
                # 发送消息给客户端
                conn.send("OK".encode())
            else:
                connected = False
        except ConnectionResetError:
            print('lose connection')
            break
        except TimeoutError:
            print('time out')

    # 关闭连接
    conn.close()


def start_server():

    # 创建TCP服务器
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 绑定IP地址和端口号
    server.bind((HOST, PORT))

    # 开始监听
    server.listen()

    print("[LISTENING] Server is listening on localhost")

    while True:
        # 等待客户端连接
        conn, addr = server.accept()

        # 创建一个新线程来处理客户端的请求
        future = executor.submit(handle_client, conn, addr)

        # client_thread = threading.Thread(target=handle_client, args=(conn, addr))
        # client_thread.start()

        # 打印当前连接的客户端数量
        print(f"[ACTIVE CONNECTIONS] {executor._work_queue.qsize()}")


# 创建io stream class


class MyStream(io.IOBase):
    def __init__(self):
        self.buffer = bytearray()

    def write(self, b):
        self.buffer.extend(b)

    def read(self, n=None):
        if n is None:
            result = self.buffer
            self.buffer = bytearray()
        else:
            result = self.buffer[:n]
            self.buffer = self.buffer[n:]
        return result


class oneTCPserver(io.IOBase):
    def __init__(self, HOST=HOST, PORT=PORT) -> None:
        # 创建TCP服务器
        self._start_server(HOST, PORT)
        self.buffer = bytearray()
        self.__runnning=True
        pass
    def __del__(self) -> None:
        self.__runnning=False
        del self.buffer
        return super().__del__()
    def _start_server(self, HOST, PORT):
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 绑定IP地址和端口号
        self._server.bind((HOST, PORT))
        # 开始监听
        self._server.listen()
        # print("[LISTENING] Server is listening on localhost")
        return None

    def _get_client(self):
        self.conn, self.addr = self._server.accept()
        return self.conn, self.addr

    def loop(self):
        while(self.__runnning):
            self._get_client()
            print(f"[NEW CONNECTION] {self.addr} connected.")
            self.conn.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            self.connected = True
            while self.connected:
                # 接收客户端发送的消息
                try:
                    message = self.conn.recv(1024)
                    if message:
                        # print(f"[{self.addr}] {message.decode()}")
                        # 发送消息给客户端
                        self.buffer.extend(message)
                        # self.conn.send("OK\n".encode())
                    # else:
                        # connected = False
                except ConnectionResetError:
                    print('lose connection')
                    break
                except TimeoutError:
                    print('time out')
            # 关闭连接
            self.conn.close()
        return None

    def write(self, message: bytes):
        try:
            return self.conn.sendall(message,socket.TCP_NODELAY)
        except OSError:
            self.connected = False
            print('tcp os err')
            return -1
        except AttributeError:
            print('tcp has not connect')
            return -1

    def read(self, n=None):
        if n is None:
            result = self.buffer
            self.buffer = bytearray()
        else:
            result = self.buffer[:n]
            self.buffer = self.buffer[n:]
        return result


if __name__ == '__main__':
    buff = MyStream()
    # 创建 ThreadPoolExecutor 对象
    executor = concurrent.futures.ThreadPoolExecutor()
    my_TCP = oneTCPserver()

    executor.submit(my_TCP.loop)

    while True:
        print(my_TCP.read().decode(),end='')
        my_TCP.write('fuck you\n'.encode())

        sleep(1)
