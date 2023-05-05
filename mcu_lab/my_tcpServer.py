import socket

HOST = '127.0.0.1'  # 服务器IP地址
PORT = 8888        # 服务器端口号

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
import socket
import threading

def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")

    connected = True
    while connected:
        # 接收客户端发送的消息
        message = conn.recv(1024)
        if message:
            print(f"[{addr}] {message.decode()}")
            # 发送消息给客户端
            conn.send("Message received".encode())
        else:
            connected = False

    # 关闭连接
    conn.close()

def start_server():
    # 创建TCP服务器
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 绑定IP地址和端口号
    server.bind(('localhost', 9999))

    # 开始监听
    server.listen()

    print("[LISTENING] Server is listening on localhost")

    while True:
        # 等待客户端连接
        conn, addr = server.accept()

        # 创建一个新线程来处理客户端的请求
        client_thread = threading.Thread(target=handle_client, args=(conn, addr))
        client_thread.start()

        # 打印当前连接的客户端数量
        print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")

if __name__ == '__main__':
    start_server()
