from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2RoughCfg( LeggedRobotCfg ):    # √
    class init_state( LeggedRobotCfg.init_state ):  # √
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]  # 左前腿机身关节
            'RL_hip_joint': 0.1,   # [rad]  # 左后腿机身关节
            'FR_hip_joint': -0.1 ,  # [rad] # 右前腿机身关节
            'RR_hip_joint': -0.1,   # [rad] # 右后腿机身关节

            'FL_thigh_joint': 0.8,     # [rad]  # 左前腿大腿关节
            'RL_thigh_joint': 1.,   # [rad] # 左后腿大腿关节
            'FR_thigh_joint': 0.8,     # [rad]  # 右前腿大腿关节
            'RR_thigh_joint': 1.,   # [rad] # 右后腿大腿关节

            'FL_calf_joint': -1.5,   # [rad]    # 左前腿小腿关节
            'RL_calf_joint': -1.5,    # [rad]   # 左后腿小腿关节
            'FR_calf_joint': -1.5,  # [rad] # 右前腿小腿关节
            'RR_calf_joint': -1.5,    # [rad]   # 右后腿小腿关节
        }

    class control( LeggedRobotCfg.control ):    # √
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        # 动作缩放：目标角度 = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):    # √
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):    # √
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0

        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

class GO2RoughCfgPPO( LeggedRobotCfgPPO ):  # √
    class algorithm( LeggedRobotCfgPPO.algorithm ): # √
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):   # √
        run_name = ''
        experiment_name = 'test_go2'
