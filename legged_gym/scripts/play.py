from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    # 重写用于测试的一些参数
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    # 环境数量
    env_cfg.terrain.num_rows = 5
    # 地形行数
    env_cfg.terrain.num_cols = 5
    # 地形列数
    env_cfg.terrain.curriculum = False
    # 关卡是否是逐渐增加难度的（即课程学习）
    env_cfg.noise.add_noise = False
    # 是否添加噪声
    env_cfg.domain_rand.randomize_friction = False
    # 是否随机化地形摩擦系数
    env_cfg.domain_rand.push_robots = False
    # 是否随机推撞机器人

    # prepare environment
    # 准备环境
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    # 加载策略
    train_cfg.runner.resume = True
    # 继续训练？
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    # ？
    
    # export policy as a jit module (used to run it from C++)
    # 导出策略为jit模块（用于在C++环境中运行）
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    # 机器人索引，用于选择记录哪个机器人的数据
    joint_index = 1 # which joint is used for logging
    # 机器人关节索引，用于选择记录哪个关节的数据
    stop_state_log = 100 # number of steps before plotting states
    # 记录状态的步数
    # 绘制状态之前的步数
    # 注意：这里的步数指的是环境步数，而不是训练步数（GPT生成）
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    # 记录奖励的步数
    # 打印回合平均奖励之前的步数
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    # 相机位置
    camera_vel = np.array([1., 1., 0.])
    # 相机速度
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    # 相机方向
    # 注意：这里的相机位置、速度、方向都是相对于世界坐标系的（GPT生成）
    # 绘制图像之前，需要设置相机位置、速度、方向（GPT生成）
    img_idx = 0
    # 图像索引

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())  # *
        obs, _, rews, dones, infos = env.step(actions.detach()) # *
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    # 目标关节位置
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    # 当前关节位置
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    # 当前关节速度
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    # 当前关节力矩
                    'command_x': env.commands[robot_index, 0].item(),
                    # x方向线速度指令
                    'command_y': env.commands[robot_index, 1].item(),
                    # y方向线速度指令
                    'command_yaw': env.commands[robot_index, 2].item(),
                    # z轴角速度指令
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    # x方向基座速度
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    # y方向基座速度
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    # z方向基座速度
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    # 基座z轴角速度
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                    # 足端接触力
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = True
    # 是否导出策略为jit脚本，用于在C++环境中运行
    RECORD_FRAMES = False
    # 是否记录每一步的图像，用于视频生成
    # 注意：如果设置为True，需要安装isaacgym的可视化功能，并设置env_cfg.viewer.enable=True
    MOVE_CAMERA = True
    # 是否在可视化界面中移动相机，用于观察动作效果
    args = get_args()
    play(args)
