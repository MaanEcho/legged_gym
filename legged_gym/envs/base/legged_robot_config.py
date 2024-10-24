from .base_config import BaseConfig


class LeggedRobotCfg(BaseConfig):   # 大部分理解了
    class env:  # √
        num_envs = 4096 # 环境数量
        num_observations = 235  # 观测空间维度
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        # 特权观测空间维度
        num_actions = 12    # 动作空间维度
        env_spacing = 3.  # not used with heightfields/trimeshes 
        # 环境间隔，单位：米
        send_timeouts = True # send time out information to the algorithm
        # 是否发送超时信息到算法
        episode_length_s = 20 # episode length in seconds
        # 单个episode的长度，单位：秒

    class terrain:  # 没完全理解参数含义
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True   # 是否应用课程学习方法
        static_friction = 1.0   # 静摩擦力
        dynamic_friction = 1.0  # 动摩擦力
        restitution = 0.    # 恢复系数

        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # 总结：等级X类型
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]

        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class commands: # √
        curriculum = False   # 是否应用课程学习方法
        max_curriculum = 1. # 最大课程?
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        # 命令数量
        resampling_time = 10. # time before command are changed[s]
        # 命令重采样时间，单位：秒
        # 重采样时间内，命令将保持不变
        heading_command = True # if true: compute ang vel command from heading error
        # 是否使用航向角作为命令
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0] # min max [m/s]
            ang_vel_yaw = [-1, 1]   # min max [rad/s]
            heading = [-3.14, 3.14] # min max [rad]

    class init_state:   # √
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        # 起始位置，单位：米
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        # 起始姿态，单位：四元数
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        # 初始线速度，单位：米/秒
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        # 初始角速度，单位：弧度/秒
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}
        # 默认关节角度，单位：弧度
        # 当action=0.0时，目标角度：{"joint_a": 0., "joint_b": 0.}

    class control:  # 不知道decimation的含义
        control_type = 'P' # P: position, V: velocity, T: torques
        # 控制类型，可选值：'P'(位置)，'V'(速度)，'T'(力矩)
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        # 刚度，单位：牛顿/弧度
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # 阻尼，单位：牛顿*秒/弧度
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:    # 没完全理解参数含义
        file = ""   # 模型文件路径
        name = "legged_robot"  # actor name
        # 模型名称
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        # 足端名称，用于索引足端状态和接触力张量
        penalize_contacts_on = []  # 惩罚相互接触的身体部位名称
        terminate_after_contacts_on = []    # 终止相互接触的身体部位名称
        disable_gravity = False  # 禁止重力
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fix the base of the robot
        # 机器人基座是否固定
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        # 默认关节驱动模式
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        # 是否允许自身碰撞
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        # 碰撞模型是否用胶囊代替圆柱体（这可以提高模拟速度和稳定性）
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001  # 密度，单位：千克/米^3
        angular_damping = 0.    # 角阻尼
        linear_damping = 0.    # 线性阻尼
        max_angular_velocity = 1000.    # 最大角速度，单位：弧度/秒
        max_linear_velocity = 1000. # 最大线速度，单位：米/秒
        armature = 0.   # Isaac Gym文档说明：为资产的所有刚体/连杆的惯性张量的对角元素添加的值。可以提高仿真稳定性
        thickness = 0.01  # 厚度，单位：米

    class domain_rand:  # √
        randomize_friction = True   # 是否随机化摩擦系数
        friction_range = [0.5, 1.25]    # 摩擦系数范围
        randomize_base_mass = False # 是否随机化基座质量
        added_mass_range = [-1., 1.]    # 附加质量范围
        push_robots = True  # 是否随机推撞机器人
        push_interval_s = 15    # 推撞间隔，单位：秒
        max_push_vel_xy = 1.    # 最大推撞速度（x轴、y轴），单位：米/秒

    class rewards:  # √
        class scales:
            termination = -0.01 # 回合终止
            tracking_lin_vel = 1.0  # 跟踪线速度
            tracking_ang_vel = 0.5  # 跟踪角速度
            lin_vel_z = -2.0    # z轴线速度
            ang_vel_xy = -0.05  # x轴、y轴角速度
            orientation = -0.   # 姿态
            torques = -0.00001  # 扭矩
            dof_vel = -0.   # 关节速度
            dof_acc = -2.5e-7   # 关节加速度
            base_height = -0.   # 基座高度
            feet_air_time =  1.0    # 足端悬空时间
            collision = -1. # 碰撞
            feet_stumble = -0.0 # 脚部晃动/绊倒
            action_rate = -0.01 # 动作频率
            stand_still = -0.   # 站立

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        # 是否只采用正奖励
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        # 跟踪奖励=exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        # 软DOF位置限制，单位：百分比，超过此限制的DOF值将被惩罚
        soft_dof_vel_limit = 1.
        # 软DOF速度限制，单位：百分比，超过此限制的DOF值将被惩罚
        soft_torque_limit = 1.
        # 软扭矩限制，单位：百分比，超过此限制的扭矩值将被惩罚
        base_height_target = 1.
        # 基座目标高度，单位：米
        max_contact_force = 100. # forces above this value are penalized
        # 最大接触力，单位：牛顿

    class normalization:    # √
        class obs_scales:
            lin_vel = 2.0   # 线速度
            ang_vel = 0.25   # 角速度
            dof_pos = 1.0   # 关节位置
            dof_vel = 0.05  # 关节速度
            height_measurements = 5.0   # 高度测量值
        clip_observations = 100.    # 观测截断值
        clip_actions = 100. # 动作截断值

    class noise:    # √
        add_noise = True  # 是否添加噪声
        noise_level = 1.0 # scales other values
        # 噪声等级，用于缩放其他值
        class noise_scales:
            dof_pos = 0.01  # 关节位置
            dof_vel = 1.5   # 关节速度
            lin_vel = 0.1   # 线速度
            ang_vel = 0.2   # 角速度
            gravity = 0.05  # 重力
            height_measurements = 0.1   # 高度测量值

    # viewer camera:
    class viewer:   # √
        ref_env = 0 # 参考环境
        pos = [10, 0, 6]  # [m]
        # 位置，单位：米
        lookat = [11., 5, 3.]  # [m]
        # 观察方向，单位：米

    class sim:  # √
        dt =  0.005 # 仿真步长，单位：秒
        substeps = 1  # 子步数
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        # 重力加速度，单位：米/秒^2
        up_axis = 1  # 0 is y, 1 is z
        # 竖直轴

        class physx:
            num_threads = 10    # 线程数
            solver_type = 1  # 0: pgs, 1: tgs
            # 解算器类型
            num_position_iterations = 4 # 位置迭代次数
            num_velocity_iterations = 0 # 速度迭代次数
            contact_offset = 0.01  # [m]
            # 接触偏移，单位：米
            rest_offset = 0.0   # [m]
            # 静止偏移，单位：米
            bounce_threshold_velocity = 0.5 # 0.5 [m/s]
            # 触发反弹所需的相对速度，单位：米/秒
            max_depenetration_velocity = 1.0
            # 求解器为纠正接触中的穿透而允许引入的最大速度，单位：米/秒
            max_gpu_contact_pairs = 2**23 # 2**24 -> needed for 8000 envs and more
            # 最大GPU接触对数
            default_buffer_size_multiplier = 5
            # 默认缓冲区大小乘数，默认值：5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            # 接触收集模式

class LeggedRobotCfgPPO(BaseConfig):    # 大部分理解了
    seed = 1    # 随机种子
    runner_class_name = 'OnPolicyRunner'  # 运行器类名
    class policy:   # √
        init_noise_std = 1.0    # 初始噪声标准差
        actor_hidden_dims = [512, 256, 128] # actor网络隐藏层维度
        critic_hidden_dims = [512, 256, 128]    # critic网络隐藏层维度
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:    # 没完全理解参数含义
        # training params
        value_loss_coef = 1.0   # 值函数损失系数
        use_clipped_value_loss = True   # 是否使用裁剪的值函数损失，以限制更新的步幅
        clip_param = 0.2    # 裁剪参数，用于限制策略更新中的概率比率(surrogate loss)变化
        entropy_coef = 0.01 # 熵系数，用于鼓励策略的探索(增加不确定性)
        num_learning_epochs = 5 # 每个训练周期中进行学习更新的次数(GPT生成)
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nmini_batches
        # 每个学习周期划分为的小批次数据数量(GPT生成)
        learning_rate = 1.e-3 # 5.e-4
        # 学习率
        schedule = 'adaptive' # could be adaptive, fixed
        # 学习率调度策略
        gamma = 0.99    # 折扣因子
        lam = 0.95  # 替代优势(GAE)计算中的平衡系数
        desired_kl = 0.01   # 期望的KL散度，用于监控训练过程中过大的策略改变
        max_grad_norm = 1.  # 梯度剪裁的最大范数，以控制梯度更新的大小

    class runner:   # √
        policy_class_name = 'ActorCritic'   # 策略类名
        algorithm_class_name = 'PPO'    # 算法类名
        num_steps_per_env = 24 # per iteration
        # 每个环境中，每次迭代进行的步数
        max_iterations = 1500 # number of policy updates
        # 策略最大迭代次数

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        # 保存间隔，单位：迭代次数
        experiment_name = 'test'    # 实验名称
        run_name = ''   # 运行名称
        # load and resume
        resume = False  # 是否从以前的检查点恢复训练
        load_run = -1 # -1 = last run
        # 当resume=True时要加载的运行的名称。如果是-1，则加载最后一个运行。
        checkpoint = -1 # -1 = last saved model
        # 要加载的检查点的编号。如果是-1，则加载最后一个保存的模型。
        resume_path = None # updated from load_run and chkpt
        # 模型文件恢复路径