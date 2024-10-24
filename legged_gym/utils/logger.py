"""
在Python项目中，logger.py 文件通常是用来配置和实现日志记录功能的。日志记录（Logging）是一种监控和调试应用程序的重要手段，它可以帮助开发者跟踪程序的运行状态，诊断问题，以及在生产环境中监控系统的健康状况。
"""

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value

class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        # defaultdict(list) 实例化类defaultdict()，并创建一个字典self.state_log。当访问字典self.state_log中不存在的键时，defaultdict(list)会自动创建一个空列表，而不是报告错误。
        self.rew_log = defaultdict(list)
        # defaultdict(list) 实例化类defaultdict()，并创建一个字典self.rew_log。当访问字典self.rew_log中不存在的键时，defaultdict(list)会自动创建一个空列表，而不是报告错误。
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        # 删除所有键值对，清空字典
        self.rew_log.clear()
        # 删除所有键值对，清空字典

    def plot_states(self):
        """
        这段代码的主要功能是启动一个子进程来执行 _plot 方法，该方法负责绘制状态日志。通过使用子进程，可以确保绘图操作不会阻塞主进程的执行，从而提高程序的并发性和响应性。

        具体来说，这段代码的主要步骤包括：

        创建一个子进程，指定 _plot 方法作为子进程的执行目标。
        启动子进程，使其开始执行 _plot 方法中的代码。
        通过这种方式，plot_states 方法实现了异步绘制状态日志的功能，确保主进程可以继续执行其他任务而不被绘图操作阻塞。
        """
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def _plot(self):
        nb_rows = 3
        # 定义子图行数
        nb_cols = 3
        # 定义子图列数
        fig, axs = plt.subplots(nb_rows, nb_cols)
        # 创建一个包含(nb_rows X nb_cols)个子图的图形窗口，并返回图形对象fig和子图对象数组axs。
        fig.tight_layout()
        # 自动调整子图间距

        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log = self.state_log

        # plot joint targets and measured positions
        # 绘制关节目标位置和测量位置
        a = axs[1, 0]
        if log["dof_pos"]: a.plot(time, log["dof_pos"], label='measured')
        # 这句代码的主要功能是检查字典 log 中是否存在键 "dof_pos" 并且该键对应的值不为空。如果条件满足，则在指定的子图 a 上绘制时间与关节位置的关系图，并为其添加标签 'measured'，以便在图例中显示。
        if log["dof_pos_target"]: a.plot(time, log["dof_pos_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position')
        a.legend()
        # plot joint velocity
        # 绘制关节速度
        a = axs[1, 1]
        if log["dof_vel"]: a.plot(time, log["dof_vel"], label='measured')
        if log["dof_vel_target"]: a.plot(time, log["dof_vel_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        a.legend()
        # plot base vel x
        # 绘制机器人x方向线速度
        a = axs[0, 0]
        if log["base_vel_x"]: a.plot(time, log["base_vel_x"], label='measured')
        if log["command_x"]: a.plot(time, log["command_x"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
        a.legend()
        # plot base vel y
        # 绘制机器人y方向线速度
        a = axs[0, 1]
        if log["base_vel_y"]: a.plot(time, log["base_vel_y"], label='measured')
        if log["command_y"]: a.plot(time, log["command_y"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
        a.legend()
        # plot base vel yaw
        # 绘制机器人z轴角速度
        a = axs[0, 2]
        if log["base_vel_yaw"]: a.plot(time, log["base_vel_yaw"], label='measured')
        if log["command_yaw"]: a.plot(time, log["command_yaw"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base velocity yaw')
        a.legend()
        # plot base vel z
        # 绘制机器人z轴线速度
        a = axs[1, 2]
        if log["base_vel_z"]: a.plot(time, log["base_vel_z"], label='measured')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity z')
        a.legend()
        # plot contact forces
        # 绘制接触力
        a = axs[2, 0]
        if log["contact_forces_z"]:
            forces = np.array(log["contact_forces_z"])
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f'force {i}')
        a.set(xlabel='time [s]', ylabel='Forces z [N]', title='Vertical Contact forces')
        a.legend()
        # plot torque/vel curves
        # 绘制扭矩/速度曲线
        a = axs[2, 1]
        if log["dof_vel"]!=[] and log["dof_torque"]!=[]: a.plot(log["dof_vel"], log["dof_torque"], 'x', label='measured')
        a.set(xlabel='Joint vel [rad/s]', ylabel='Joint Torque [Nm]', title='Torque/velocity curves')
        a.legend()
        # plot torques
        # 绘制扭矩
        a = axs[2, 2]
        if log["dof_torque"]!=[]: a.plot(time, log["dof_torque"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        a.legend()

        plt.show()

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        """
        这段代码的主要功能是在 Logger 类的实例被销毁时，确保相关的子进程（如果有）被正确终止。这样可以避免子进程在后台继续运行，占用系统资源。通过在析构函数中添加这一逻辑，确保了资源的正确释放和清理。
        __del__ 是 Python 中的一个特殊方法，用于在对象被销毁（即垃圾回收时）时执行一些清理操作。
        """
        if self.plot_process is not None:
        # 如果 self.plot_process 不是 None，说明子进程正在运行或已经启动。
            self.plot_process.kill()