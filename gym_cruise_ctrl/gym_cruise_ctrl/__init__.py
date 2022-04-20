from gym.envs.registration import register

register(id='cruise-ctrl-v0', entry_point='gym_cruise_ctrl.envs:CruiseCtrlEnv',)