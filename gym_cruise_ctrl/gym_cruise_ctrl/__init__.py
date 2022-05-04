from gym.envs.registration import register

register(id='cruise-ctrl-v0', entry_point='gym_cruise_ctrl.envs:CruiseCtrlEnv',)
register(id='cruise-ctrl-v1', entry_point='gym_cruise_ctrl.envs:CruiseCtrlEnv1',)
register(id='cruise-ctrl-v2', entry_point='gym_cruise_ctrl.envs:CruiseCtrlEnv2',)