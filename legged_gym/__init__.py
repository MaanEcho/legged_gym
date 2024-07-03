import os

LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
"""
这行代码的主要功能是获取 legged_gym 项目的根目录路径。通过两次调用 os.path.dirname()，代码从当前脚本的路径逐步向上追溯，最终得到项目的根目录路径，并将其赋值给 LEGGED_GYM_ROOT_DIR 变量。
"""
LEGGED_GYM_ENVS_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, 'legged_gym', 'envs')
"""
这行代码的主要功能是生成并存储 legged_gym 项目中 envs 目录的完整路径。通过使用 os.path.join() 函数，代码将根目录路径与子目录名称拼接起来，确保生成的路径在不同操作系统下都是正确的。最终，这个路径被赋值给 LEGGED_GYM_ENVS_DIR 变量，方便后续代码使用。
"""