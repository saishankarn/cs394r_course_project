from gym.envs.registration import register

register(id='cruise-ctrl-v0', entry_point='gym_cruise_ctrl.envs:CruiseCtrlEnv',)
#register(id='cruise-ctrl-v1', entry_point='gym_cruise_ctrl.envs:CruiseCtrlEnv1',)
register(id='cruise-ctrl-v2', entry_point='gym_cruise_ctrl.envs:CruiseCtrlEnv2',)
register(id='cruise-ctrl-v3', entry_point='gym_cruise_ctrl.envs:CruiseCtrlEnv3',)
register(id='cruise-ctrl-v4', entry_point='gym_cruise_ctrl.envs:CruiseCtrlEnv4',)
register(id='cruise-ctrl-v5', entry_point='gym_cruise_ctrl.envs:CruiseCtrlEnv2D',)
register(id='cruise-ctrl-v6', entry_point='gym_cruise_ctrl.envs:CruiseCtrlEnv2DRandom',)
register(id='cruise-ctrl-v7', entry_point='gym_cruise_ctrl.envs:CruiseCtrlEnvSc',)