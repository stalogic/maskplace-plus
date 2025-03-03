import binascii
import os
import time
import numpy as np


def is_gzip_file(file_path):
    with open(file_path, 'rb') as file:
        header = file.read(2)
    hex_header = binascii.hexlify(header).decode('utf-8')
    if hex_header == '1f8b':
        return True
    else:
        return False
    

def my_range(start, end):
    if start == end:
        return [start]
    if start != end:
        return range(start, end)
    

def instance_direction_rect(line):  # used when we only need bounding box (rect) of the cell.
    if 'N' in line or 'S' in line:
        m_direction = (1, 0, 0, 1)
    elif 'W' in line or 'E' in line:
        m_direction = (0, 1, 1, 0)
    else:
        raise ValueError('read_macro_direction_wrong')
    return m_direction


def save_numpy(root_path, dir_name, save_name, data):
    save_path = os.path.join(root_path, dir_name, save_name)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    np.save(save_path, data)


def timeit(func):
    def wrapper(*args, **kwargs):
        # 记录函数开始执行的时间
        start_time = time.time()
        # 调用原始函数
        result = func(*args, **kwargs)
        # 记录函数结束执行的时间
        end_time = time.time()
        # 计算函数执行的时间
        execution_time = end_time - start_time
        print(f"函数 {func.__name__} 执行时间: {execution_time} 秒")
        return result
    return wrapper

