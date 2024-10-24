import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize
from typing import Tuple

# @ torch.jit.script
"""
@ torch.jit.script 装饰器的主要功能是将 Python 函数编译成 TorchScript，从而提高执行效率并在不依赖 Python 解释器的环境中运行。通过这种方式，可以优化代码的性能，并使其能够在更广泛的场景中使用，例如在 C++ 环境中。
"""
def quat_apply_yaw(quat, vec):
    """
    这段代码的主要功能是使用输入的四元数 quat 对向量 vec 进行偏航（yaw）旋转。具体来说，它通过将四元数的 x 和 y 分量设置为 0，只保留 z 轴上的旋转部分，然后对向量进行旋转操作。这种方法在机器人学和计算机图形学中常用于处理三维空间中的旋转问题，特别是在需要分离出特定轴向旋转的场景中。
    """
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    # 将 quat_yaw 的前两个元素（即四元数的 x 和 y 分量）设置为 0。这实际上是将四元数的旋转部分限制在 z 轴上，即只保留偏航（yaw）旋转。
    quat_yaw = normalize(quat_yaw)
    # 对修改后的四元数 quat_yaw 进行归一化处理，确保其仍然是一个有效的单位四元数。
    return quat_apply(quat_yaw, vec)
    # 使用归一化后的四元数 quat_yaw 对输入向量 vec 进行旋转操作，并返回旋转后的向量。

# @ torch.jit.script
def wrap_to_pi(angles):
    """
    这段代码的主要功能是将输入的角度 angles 调整到 -π 到 π 的范围内。具体步骤包括：
    1、将角度取模 2π，确保角度在 0 到 2π 之间。
    2、将大于 π 的角度减去 2π，使其落入 -π 到 π 的范围内。
    这种处理在机器人学、计算机图形学和物理模拟中非常常见，特别是在需要标准化角度以简化计算和避免数值问题的场景中。
    """
    angles %= 2*np.pi
    # 在这个表达式中，angles 与 2*np.pi 进行取模运算，意味着 angles 的值会被调整到介于 0 和 2*np.pi 之间（包括 0 但不包括 2*np.pi）。这是因为一个完整的圆的角度表示为 2*np.pi 弧度，所以任何大于 2*np.pi 的角度都可以通过减去若干个 2*np.pi 来转换为一个等效的、介于 0 到 2*np.pi 之间的角度。
    angles -= 2*np.pi * (angles > np.pi)
    """
    这行代码的作用是将角度调整到 -π 到 π 的范围内。具体来说：
    1、angles > np.pi 会生成一个布尔数组，表示哪些角度大于 π。
    2、2*np.pi * (angles > np.pi) 会生成一个数组，其中大于 π 的角度对应的位置是 2π，其他位置是 0。
    3、angles -= 2*np.pi * (angles > np.pi) 会将大于 π 的角度减去 2π，从而将这些角度调整到 -π 到 0 的范围内。
    """
    return angles

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    """
    这段代码的主要功能是生成一个形状为 shape 的随机数张量，并将其映射到指定的范围 [lower, upper) 之间。生成的随机数具有非线性分布特性，通过平方根变换实现。具体步骤包括：
    1、生成范围在 [-1, 1) 之间的随机数张量。
    2、对随机数张量进行平方根变换，使其具有非线性分布特性。
    3、将变换后的随机数张量调整到 [0, 1) 之间。
    4、将调整后的随机数张量映射到指定的范围 [lower, upper) 之间。
    5、这种非线性分布的随机数生成在某些特定的模拟和优化问题中非常有用，可以更好地模拟现实世界中的复杂分布情况。
    """
    # type: (float, float, Tuple[int, int], str) -> Tensor
    # 类型：（浮点数，浮点数，元组[整型，整型]，字符串）-> 张量
    r = 2*torch.rand(*shape, device=device) - 1
    """
    这行代码生成一个形状为 shape 的随机数张量，范围在 [0, 1) 之间。然后通过乘以 2 并减去 1，将范围调整到 [-1, 1) 之间。
    torch.rand 是 PyTorch 中的一个函数，用于生成指定形状的随机数张量，范围在 [0, 1) 之间。
    *shape 是一个参数解包（unpacking）的操作。*shape 将元组 shape 解包成两个独立的参数，例如如果 shape 是 (3, 4)，那么 *shape 就相当于 3, 4。
    device=device 指定了生成的张量所在的设备（例如 CPU 或 GPU）。
    """
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    """
    这行代码对生成的随机数张量 r 进行平方根变换。具体来说：
    1、如果 r 小于 0，取 -torch.sqrt(-r)，即对负数取平方根并取负。
    2、如果 r 大于或等于 0，取 torch.sqrt(r)，即对正数取平方根。 这种变换使得生成的随机数在 [-1, 1) 范围内具有非线性分布特性。
    """
    r =  (r + 1.) / 2.
    # 这行代码将变换后的随机数张量 r 的范围从 [-1, 1) 调整到 [0, 1) 之间。
    return (upper - lower) * r + lower