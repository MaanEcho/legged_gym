from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        # 初始位置
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            # 左前腿机身关节
            'RL_hip_joint': 0.1,   # [rad]
            # 左后腿机身关节
            'FR_hip_joint': -0.1 ,  # [rad]
            # 右前腿机身关节
            'RR_hip_joint': -0.1,   # [rad]
            # 右后腿机身关节

            'FL_thigh_joint': 0.8,     # [rad]
            # 左前腿大腿关节
            'RL_thigh_joint': 1.,   # [rad]
            # 左后腿大腿关节
            'FR_thigh_joint': 0.8,     # [rad]
            # 右前腿大腿关节
            'RR_thigh_joint': 1.,   # [rad]
            # 右后腿大腿关节

            'FL_calf_joint': -1.5,   # [rad]
            # 左前腿小腿关节
            'RL_calf_joint': -1.5,    # [rad]
            # 左后腿小腿关节
            'FR_calf_joint': -1.5,  # [rad]
            # 右前腿小腿关节
            'RR_calf_joint': -1.5,    # [rad]
            # 右后腿小腿关节
        }
        # 默认关节角度，单位：弧度
        # 当action=0时，目标角度为default_joint_angles

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        # PD控制参数：
        control_type = 'P'
        # 控制类型
        stiffness = {'joint': 20.}  # [N*m/rad]
        # 刚度，单位：N*m/rad
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # 阻尼，单位：N*m*s/rad
        # action scale: target angle = actionScale * action + defaultAngle
        # 动作缩放：目标角度 = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        # decimation(降采样比)：在每个策略决策时间（policy DT）内，模拟时间（sim DT）中控制动作更新的次数。
        """
        "Decimation" 在此语境下指的是：在每个策略决策时间（policy DT）内，模拟时间（sim DT）中控制动作更新的次数。简而言之，它描述了控制指令在模拟环境中的更新频率与策略制定周期之间的比例关系。
        例如，如果 decimation 设置为 5，则意味着在策略周期的每一单位时间内，控制动作将被更新 5 次。这通常用于调整模型预测控制（MPC）或类似控制系统中计算资源和实时性能之间的平衡。
        """
        decimation = 4

    class asset( LeggedRobotCfg.asset ):    # 机器人模型
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        # 机器人URDF文件路径
        name = "go2"
        # 名称
        foot_name = "foot"
        # 脚部名称
        penalize_contacts_on = ["thigh", "calf"]
        # 接触惩罚的身体部位名称列表
        terminate_after_contacts_on = ["base"]
        # 接触终止的身体部位名称列表
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        # 自身碰撞？
        # 1：禁止，0：允许...位掩码过滤器
  
    class rewards( LeggedRobotCfg.rewards ):
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            # 扭矩惩罚系数
            dof_pos_limits = -10.0
            # DOF位置限制惩罚系数

        soft_dof_pos_limit = 0.9
        # 软DOF位置限制，单位：百分比，超过此限制的DOF值将被惩罚
        base_height_target = 0.25
        # 基座目标高度，单位：米？

class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        # 熵权系数
    class runner( LeggedRobotCfgPPO.runner ):   # 运行器
        run_name = 'test'
        # 运行名称
        experiment_name = 'test_go2'
        # 实验名称
