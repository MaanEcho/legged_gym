# A config file
from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):   # 包含所有的环境参数
    class env:  # 环境
        num_envs = 4096
        # 环境数量
        num_observations = 235
        # 观测空间维度
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        # 特权观测空间维度
        # 如果num_privileged_obs不为None，则在step()函数中返回特权观测空间priviledge_obs_buf（用于不对称训练的critic观测）。否则，返回None。
        num_actions = 12
        # 动作空间维度
        env_spacing = 3.  # not used with heightfields/trimeshes 
        # 环境间隔，单位：米？
        # 地形为heightfield或trimesh时，此参数无效(不使用)
        send_timeouts = True # send time out information to the algorithm
        # 是否发送超时信息到算法
        # 发送超时信息到算法，以便在超时时终止episode
        episode_length_s = 20 # episode length in seconds
        # 单个episode的长度，单位：秒

    class terrain:  # 地形
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        # 地形网格类型，可选值：'none'(无地形)，'plane'(平面)，'heightfield'(高度场)，'trimesh'。
        horizontal_scale = 0.1 # [m]
        # 水平缩放比例，单位：米
        vertical_scale = 0.005 # [m]
        # 垂直缩放比例，单位：米
        border_size = 25 # [m]
        # 边界大小，单位：米
        curriculum = True
        # 是否应用课程学习方法
        static_friction = 1.0
        # 静态摩擦系数
        dynamic_friction = 1.0
        # 动态摩擦系数
        restitution = 0.
        # 恢复系数

        # rough terrain only:
        # 仅粗糙地形适用：
        measure_heights = True
        # 是否测量高度
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        # 高度测量点在x方向上的相对位置，相对于地形的中心线
        # 1mx1.6m矩形（不含中心线）
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        # 高度测量点在y方向上的相对位置，相对于地形的中心线
        # 1mx1.6m矩形（不含中心线）
        selected = False # select a unique terrain type and pass all arguments
        # 是否选择特定的地形类型并传递所有参数
        # 选择特定的地形类型并传递所有参数
        terrain_kwargs = None # Dict of arguments for selected terrain
        # 选定的地形类型的参数字典
        max_init_terrain_level = 5 # starting curriculum state
        # 课程学习开始时，最大初始地形等级
        # 开始课程学习状态
        terrain_length = 8.
        # 地形长度，单位：米？
        terrain_width = 8.
        # 地形宽度，单位：米？
        num_rows= 10 # number of terrain rows (levels)
        # 地形行数（等级）的数量
        num_cols = 20 # number of terrain cols (types)
        # 地形列数（类型）的数量
        # 总结：等级X类型
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # 地形类型：[平滑坡度，粗糙坡度，楼梯上升，楼梯下降，离散]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # 地形比例：[平滑坡度，粗糙坡度，楼梯上升，楼梯下降，离散]

        # trimesh only:
        # 仅trimesh适用：
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
        # 坡度阈值，超过此阈值的坡度将被修正为垂直表面

    class commands: # 命令相
        curriculum = False
        # 是否应用课程学习方法
        max_curriculum = 1.
        # 最大课程?
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        # 命令数量，默认：x方向线速度，y方向线速度，z轴角速度，航向角（航向模式下，z轴角速度由航向误差计算得到）
        resampling_time = 10. # time before command are changed[s]
        # 命令重采样时间，单位：秒
        # 重采样时间内，命令将保持不变
        heading_command = True # if true: compute ang vel command from heading error
        # 是否使用航向角作为命令
        # 如果为True，则根据航向误差计算z轴角速度命令
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            # x方向线速度范围，单位：米/秒
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            # y方向线速度范围，单位：米/秒
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            # z轴角速度范围，单位：弧度/秒
            heading = [-3.14, 3.14]
            # 航向角范围，单位：弧度

    class init_state:   # 初始状态
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

    class control:  # 控制
        control_type = 'P' # P: position, V: velocity, T: torques
        # 控制类型，可选值：'P'(位置)，'V'(速度)，'T'(力矩)
        # PD Drive parameters:
        # PD控制参数：
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        # 刚度，单位：牛顿/弧度
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # 阻尼，单位：牛顿*秒/弧度
        # action scale: target angle = actionScale * action + defaultAngle
        # 动作缩放：目标角度=actionScale*action+defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        # decimation(降采样比)：在每个策略决策时间（policy DT）内，模拟时间（sim DT）中控制动作更新的次数。
        """
        "Decimation" 在此语境下指的是：在每个策略决策时间（policy DT）内，模拟时间（sim DT）中控制动作更新的次数。简而言之，它描述了控制指令在模拟环境中的更新频率与策略制定周期之间的比例关系。
        例如，如果 decimation 设置为 5，则意味着在策略周期的每一单位时间内，控制动作将被更新 5 次。这通常用于调整模型预测控制（MPC）或类似控制系统中计算资源和实时性能之间的平衡。
        """
        decimation = 4

    class asset:    # 机器人模型
        file = ""
        # 模型文件路径
        name = "legged_robot"  # actor name
        # 名称
        # 演员名称
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        # 脚部名称，用于索引脚部状态和接触力张量
        penalize_contacts_on = []
        # 接触惩罚的身体部位名称列表
        terminate_after_contacts_on = []
        # 接触终止的身体部位名称列表
        disable_gravity = False
        # 禁止重力
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        # 折叠固定关节？
        # 合并连接着固定关节的身体。可以通过添加“<... dont_collapse=“true”>”来保留特定的固定关节。
        fix_base_link = False # fix the base of the robot
        # 固定基座链接？
        # 固定机器人的基座。
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        # 默认关节驱动模式
        # 见GymDofDriveModeFlags（0：无，1：位置目标，2：速度目标，3：力矩）
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        # 自身碰撞？
        # 1：禁止，0：允许...位掩码过滤器
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        # 用胶囊代替圆柱体？
        # 用胶囊代替圆柱体，可以提高模拟速度和稳定性。
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        # 翻转视觉附件？
        # 一些.obj网格必须从y-up翻转到z-up。
        
        density = 0.001
        # 密度，单位：千克/米^3？
        angular_damping = 0.
        # 角阻尼，单位：牛顿/秒？
        linear_damping = 0.
        # 线性阻尼，单位：牛顿/秒？
        max_angular_velocity = 1000.
        # 最大角速度，单位：弧度/秒？
        max_linear_velocity = 1000.
        # 最大线速度，单位：米/秒？
        armature = 0.
        # 弹簧？
        thickness = 0.01
        # 厚度，单位：米？

    class domain_rand:  # 域随机化
        randomize_friction = True
        # 随机化摩擦系数
        friction_range = [0.5, 1.25]
        # 摩擦系数范围
        randomize_base_mass = False
        # 随机化基座质量
        added_mass_range = [-1., 1.]
        # 附加质量范围
        push_robots = True
        # 随机推撞机器人
        push_interval_s = 15
        # 推撞间隔，单位：秒
        max_push_vel_xy = 1.
        # 最大推撞速度（x轴、y轴），单位：米/秒？

    class rewards:  # 奖励函数
        class scales:   # 奖励函数缩放系数
            termination = -0.0
            # 回合终止
            tracking_lin_vel = 1.0
            # 跟踪线速度
            tracking_ang_vel = 0.5
            # 跟踪角速度
            lin_vel_z = -2.0
            # z轴线速度
            ang_vel_xy = -0.05
            # x轴、y轴角速度
            orientation = -0.
            # 姿态
            torques = -0.00001
            # 力矩
            dof_vel = -0.
            # 关节速度
            dof_acc = -2.5e-7
            # 关节加速度（为什么要惩罚关节加速度？）
            base_height = -0. 
            # 基座高度
            feet_air_time =  1.0
            # 脚部空气时间？
            collision = -1.
            # 碰撞
            feet_stumble = -0.0 
            # 脚部晃动/绊倒？
            action_rate = -0.01
            # 动作频率
            stand_still = -0.
            # 站立

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        # 只允许正奖励
        # 如果为True，则负总奖励将被截断为零（避免过早终止问题）
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        # 跟踪sigma（西格玛）
        # 跟踪奖励=exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        # 软DOF位置限制，单位：百分比，超过此限制的DOF值将被惩罚
        soft_dof_vel_limit = 1.
        # 软DOF速度限制，单位：百分比，超过此限制的DOF值将被惩罚
        soft_torque_limit = 1.
        # 软力矩限制，单位：百分比，超过此限制的力矩值将被惩罚
        base_height_target = 1.
        # 基座目标高度，单位：米？
        max_contact_force = 100. # forces above this value are penalized
        # 最大接触力，单位：牛顿？
        # 超过此值的接触力将被惩罚

    class normalization:    # 归一化
        class obs_scales:   # 观测值缩放系数
            lin_vel = 2.0
            # 线速度
            ang_vel = 0.25
            # 角速度
            dof_pos = 1.0
            # 关节位置
            dof_vel = 0.05
            # 关节速度
            height_measurements = 5.0
            # 高度测量值
        clip_observations = 100.
        # 观测值截断值
        clip_actions = 100.
        # 动作截断值

    class noise:    # 噪声
        add_noise = True
        # 是否添加噪声
        noise_level = 1.0 # scales other values
        # 噪声等级，用于缩放其他值
        class noise_scales: # 噪声缩放系数
            dof_pos = 0.01
            # 关节位置
            dof_vel = 1.5
            # 关节速度
            lin_vel = 0.1
            # 线速度
            ang_vel = 0.2
            # 角速度
            gravity = 0.05
            # 重力
            height_measurements = 0.1
            # 高度测量值

    # viewer camera:
    # 观察者相机
    class viewer:   # 观察者
        ref_env = 0
        # 参考环境
        pos = [10, 0, 6]  # [m]
        # 位置，单位：米
        lookat = [11., 5, 3.]  # [m]
        # 观察方向，单位：米

    class sim:  # 仿真器？
        dt =  0.005
        # 仿真步长，单位：秒？
        substeps = 1
        # 子步数？
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        # 重力加速度，单位：米/秒^2
        up_axis = 1  # 0 is y, 1 is z
        # 竖直轴
        # 0是y，1是z

        class physx:    # PhysX 物理引擎
            num_threads = 10
            # 线程数
            solver_type = 1  # 0: pgs, 1: tgs
            # 解算器类型
            # 0：pgs，1：tgs
            num_position_iterations = 4
            # 位置迭代次数
            num_velocity_iterations = 0
            # 速度迭代次数
            contact_offset = 0.01  # [m]
            # 接触偏移，单位：米
            rest_offset = 0.0   # [m]
            # 静止偏移，单位：米
            bounce_threshold_velocity = 0.5 # 0.5 [m/s]
            # 弹跳阈值速度，单位：米/秒
            max_depenetration_velocity = 1.0
            # 最大穿透恢复速度，单位：米/秒
            """
            "Max Depenetration Velocity" 可以翻译为“最大脱嵌速度”或“最大穿透恢复速度”。在物理引擎和碰撞检测算法中，这个术语指的是两个发生穿透（即重叠）的物体在解决穿透状态时的最大允许分离速度。当两个物体非法穿透（即它们的形状相交，违反了非穿透性原则）时，物理引擎会尝试将它们分开到一个合法的非穿透状态，这个过程中物体分离的速度不能超过“最大脱嵌速度”，以避免不自然的弹跳或其他不真实的行为。这个参数有助于确保模拟的稳定性和真实性。
            """
            max_gpu_contact_pairs = 2**23 # 2**24 -> needed for 8000 envs and more
            # 最大GPU接触对数，默认值：2**23
            # 对于8000个环境和更多的场景，这个值至少需要2**24。
            default_buffer_size_multiplier = 5
            # 默认缓冲区大小乘数，默认值：5
            """
            "default buffer size multiplier" 可以翻译为“默认缓冲区大小倍数”或“默认缓冲尺寸乘数”。在计算机科学和软件工程中，这通常指的是初始化缓冲区（用于暂存数据的内存区域）时，默认大小的一个乘数因子。通过调整这个乘数，可以影响初始分配的缓冲区大小，从而影响程序的性能和资源使用。例如，在网络编程、音频处理或视频流中，适当设置缓冲区大小对于避免数据丢失、减少延迟和提高用户体验至关重要。
            """
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            # 接触收集模式
            # 0：从不，1：最后一个子步，2：所有子步（默认=2）
            """
            "Contact Collection" 可以翻译为“接触收集模式”。在物理引擎中，接触收集是指物理引擎在模拟过程中收集接触信息，并将其存储在内存中，以便稍后用于计算接触力。接触收集模式有助于提高模拟效率，尤其是在复杂的场景中，其中有许多碰撞发生。
            """

class LeggedRobotCfgPPO(BaseConfig):    # 包含所有的训练参数
    seed = 1
    # 随机种子
    runner_class_name = 'OnPolicyRunner'
    # 运行器类名？
    class policy:   # 策略
        init_noise_std = 1.0
        # 初始噪声标准差
        actor_hidden_dims = [512, 256, 128]
        # 演员（策略）网络隐藏层维度
        critic_hidden_dims = [512, 256, 128]
        # 评论家（值函数）网络隐藏层维度
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # 激活函数
        # 可以是elu，relu，selu，crelu，lrelu，tanh，sigmoid

        # only for 'ActorCriticRecurrent':
        # 只适用于'ActorCriticRecurrent'：
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:    # 算法
        # training params
        # 训练参数
        value_loss_coef = 1.0
        # 值函数损失系数
        use_clipped_value_loss = True
        # 是否使用经裁剪的值函数损失
        clip_param = 0.2
        # 裁剪参数
        entropy_coef = 0.01
        # 熵损失系数
        num_learning_epochs = 5
        # 学习周期数
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nmini_batches
        # mini_batch数量
        # mini_batch大小=num_envs*nsteps/nmini_batches
        learning_rate = 1.e-3 #5.e-4
        # 学习率
        schedule = 'adaptive' # could be adaptive, fixed
        # 学习率调度器
        # 可以是adaptive或fixed
        gamma = 0.99
        # 折扣因子
        lam = 0.95
        # 加权折扣因子？
        desired_kl = 0.01
        # 期望KL？
        # 这个参数用于控制KL散度的上限，以防止策略网络过于自信而导致性能下降。
        # 如果KL散度超过期望值，则会停止更新策略网络。
        # 以上两行为GPT生成内容
        max_grad_norm = 1.
        # 最大梯度范数？
        # 这个参数用于控制梯度的最大值，以防止梯度爆炸或梯度消失。
        # 上一行内容为GPT生成内容

    class runner:   # 运行器
        policy_class_name = 'ActorCritic'
        # 策略类名
        algorithm_class_name = 'PPO'
        # 算法类名
        num_steps_per_env = 24 # per iteration
        # 每个环境的步数
        # 每次迭代的步数
        max_iterations = 1500 # number of policy updates
        # 最大迭代次数
        # 策略更新次数

        # logging
        # 日志
        save_interval = 50 # check for potential saves every this many iterations
        # 保存间隔
        # 每save_interval次迭代，检查潜在保存点
        experiment_name = 'test'
        # 实验名称
        run_name = ''
        # 运行名称
        # load and resume
        # 加载和恢复
        resume = False
        # 是否恢复
        load_run = -1 # -1 = last run
        # 加载的运行编号
        # -1表示加载最后一个运行
        checkpoint = -1 # -1 = last saved model
        # 检查点编号
        # -1表示加载最后保存的模型
        resume_path = None # updated from load_run and chkpt
        # 恢复路径
        # 由load_run和chkpt更新