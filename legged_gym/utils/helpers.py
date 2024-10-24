"""
在Python项目中，helpers.py 文件通常用于封装一系列辅助函数或者通用工具方法。这个文件的作用是提供一些跨模块使用的功能，使得代码更加模块化和可重用。helpers.py（或者类似命名如 utils.py, common.py 等）可以包含各种类型的辅助功能，包括：数据处理函数，文件操作，字符串处理，网络请求，日志记录，异常处理，装饰器，数学计算，类型检查，环境检测等。
"""

import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def class_to_dict(obj) -> dict: # √
    """
    类型转换，将一个类实例转换为字典表示
    """
    if not hasattr(obj,"__dict__"):
        return obj
    result = {} 
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    """
    这段代码的主要功能是从一个字典中更新一个对象的属性。具体来说，它遍历字典中的每个键值对，尝试将键对应的值赋给对象的相应属性。如果字典中的值本身是一个字典，它会递归地更新对象的子属性。
    （可能是在整个项目的最后阶段使用）
    """
    for key, val in dict.items():
    # 遍历字典中的每个键值对
        attr = getattr(obj, key, None)
        # 这行代码使用 getattr 函数尝试从对象 obj 中获取名为 key 的属性。如果 obj 中不存在这个属性，则返回 None。
        if isinstance(attr, type):
        # 检查 attr 是否是一个类型（即类）。
        # 如果是，则递归调用 update_class_from_dict 函数，将 attr 作为新的对象，val 作为新的字典进行更新。如果 attr 不是类型，则直接使用 setattr 函数将 obj 的 key 属性设置为 val。
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed): # √
    """设置随机种子"""
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)   # 设置 Python 内置随机数生成器的种子
    np.random.seed(seed)  # 设置 NumPy 随机数生成器的种子。
    torch.manual_seed(seed)  # 设置 PyTorch 随机数生成器的种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 设置 Python 的哈希种子。设置 PYTHONHASHSEED 环境变量可以确保哈希值的计算是确定性的。
    torch.cuda.manual_seed(seed)
    # 设置当前 CUDA 设备的随机数生成器的种子。这确保了在 GPU 上运行的 PyTorch 代码生成的随机数是确定性的。
    torch.cuda.manual_seed_all(seed)
    # 设置所有 CUDA 设备的随机数生成器的种子。这确保了所有 GPU 生成的随机数是确定性的。

def parse_sim_params(args, cfg):    # √
    """解析仿真参数"""
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1):
    """确定加载模型的路径"""
    try:
        runs = os.listdir(root)
        # 这段代码的主要功能是列出指定目录 root 中的所有文件和子目录，并将结果存储在变量 runs 中。具体来说，os.listdir(root) 函数会返回一个包含目录 root 中所有文件和子目录名称的列表，而 runs 变量则用于存储这个列表。
        # TODO: sort by date to handle change of month
        # TODO: 按日期排序以处理月份变更。
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run==-1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        # 环境数量
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def get_args(): # √
    """获取参数"""
    #----------custom parameters----------#
    custom_parameters = [
        {"name": "--task", "type": str, "default": "anymal_c_flat", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False, "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
    ]
    #----------custom parameters----------#

    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        # headless = False,
        # no_graphics = False,
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

def export_policy_as_jit(actor_critic, path):
    """
    导出策略为JIT模型
    具体来说：
    如果 actor_critic 对象包含 memory_a 属性（假设是 LSTM 模型），则使用 PolicyExporterLSTM 类来处理并导出模型。
    如果 actor_critic 对象不包含 memory_a 属性，则直接将 actor_critic.actor 模型转换为 JIT 脚本模块并保存。
    """
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        # 假设是LSTM: TODO: 添加GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        # 这句代码的主要功能是创建目录，确保指定的目录存在。如果目录不存在，它会创建这个目录及其所有必要的中间目录。如果目录已经存在，它不会引发错误，而是会静默地忽略这个操作。
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)