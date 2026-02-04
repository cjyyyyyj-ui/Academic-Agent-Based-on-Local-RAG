# src/utils.py
import os
import sys


def get_resource_path(relative_path):
    """
    适配脚本运行和exe运行的统一路径获取函数
    :param relative_path: 相对于项目根目录的文件/目录相对路径
    :return: 实际可访问的绝对路径
    """
    if hasattr(sys, '_MEIPASS'):
        # exe运行时：临时解压目录（项目根目录的所有文件都会在这里）
        base_path = sys._MEIPASS
    else:
        # 脚本运行时：项目根目录（因为main.py在根目录，启动后上下文是根目录）
        base_path = os.path.abspath(".")

    # 拼接并返回绝对路径
    return os.path.join(base_path, relative_path)