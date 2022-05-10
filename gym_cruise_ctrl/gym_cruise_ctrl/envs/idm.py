from math import sqrt

"""
### Equation from paper:
    Automated Speed and Lane Change Decision Making using Deep Reinforcement Learning
"""

class IDM():
    def __init__(self) -> None:
        self.d0    = 5      # Minimum gap distance
        self.T     = 0.0    # Safe time headway
        self.a     = 0.7    # Maximal acceleration
        self.b     = 1.5    # Desired deceleration
        self.delta = 4      # Acceleration exponent
        self.v0    = 40     # Desired velocity, also the max velocity of the ego vehicle
        self.damp  = 1.0    # Increasing the value makes the response more sluggish

    def action(self, d, delta_v, v):
        """
        ### d      : relative distance (front vehicle position - ego vehicle position)
            delta_v: approach velocity (negative of relative velocity )
            v      : ego_vehicle speed
        """
        d_star = self.d0 + v*self.T + v*delta_v/(2*sqrt(self.a*self.b))
        acc    = self.a*(1 - (v/self.v0)**self.delta - self.damp*(d_star/d)**2)
        return acc